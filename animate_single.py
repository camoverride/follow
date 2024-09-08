import cv2
import os
import numpy as np
import mediapipe as mp
import threading
import re



# Global variables
face_x_coordinate = None
face_y_coordinate = None
face_x_lock = threading.Lock()
latest_webcam_frame = None
frame_lock = threading.Lock()


MOVEMENT_SCALE = 0.8  # Scale factor to exaggerate the background movement


def detect_face_position(debug=True):
    """
    Set the (x, y) coordinates of the face in order to perform tracking.

    NOTE: if multiple faces are detected, only the coordinates for the last face are stored
    TODO: handle multi-face detection
    """
    # Global variables shared between threads
    global face_x_coordinate, face_y_coordinate, latest_webcam_frame

    # Face Detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    # Video capture from webcam.
    cap = cv2.VideoCapture(0)

    # Begin reading frames from the webcam.
    while True:
        ret, webcam_frame = cap.read()
        if not ret:
            break

        # Perform face detection
        webcam_frame_rgb = cv2.cvtColor(webcam_frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(webcam_frame_rgb)

        with face_x_lock:
            # If a face is detected, get the (x, y) coordinates.
            if results.detections:
                # There might be multiple detections...
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box

                    # Dividing by 2 gets the center of the bounding box that contains the face/
                    face_x_coordinate = bbox.xmin + bbox.width / 2
                    face_y_coordinate = bbox.ymin + bbox.height / 2

                    if debug:
                        # Draw bounding box for debugging
                        x_min = int(bbox.xmin * webcam_frame.shape[1])
                        y_min = int(bbox.ymin * webcam_frame.shape[0])
                        width = int(bbox.width * webcam_frame.shape[1])
                        height = int(bbox.height * webcam_frame.shape[0])
                        cv2.rectangle(webcam_frame, (x_min, y_min), (x_min + width, y_min + height), (0, 255, 0), 2)

            # If no face is detected, set the coordinates as None
            else:
                face_x_coordinate, face_y_coordinate = None, None

        # Update the latest frame for the main thread to access
        with frame_lock:
            latest_webcam_frame = webcam_frame.copy()

    cap.release()


def natural_sort_key(s):
    # Sort filenames that contain numbers in a "natural" way (e.g., file1, file2, file10)
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]

def images_to_memmap(image_dir, memmap_filename='images_memmap.dat'):
    """
    Takes a directory containing images and writes them all to a memmap for fast I/O
    Returns memmap_path, num_frames, frame_width, frame_height
    """
    # Get all image paths in the directory
    image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith(('png', 'jpg', 'jpeg'))]
    
    # Sort images by natural order (handles numeric filenames correctly)
    image_paths = sorted(image_paths, key=natural_sort_key)

    if not image_paths:
        raise ValueError("No images found in the specified directory.")

    # Read the first image to get dimensions
    first_image = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED)
    frame_height, frame_width = first_image.shape[:2]
    num_channels = first_image.shape[2] if len(first_image.shape) == 3 else 1  # Get the number of channels

    # Create a memory-mapped file
    num_frames = len(image_paths)
    memmap_shape = (num_frames, frame_height, frame_width, num_channels)
    memmap_path = os.path.join(image_dir, memmap_filename)
    memmap_file = np.memmap(memmap_path, dtype='uint8', mode='w+', shape=memmap_shape)

    # Load each image into the memmap in sorted order
    for i, img_path in enumerate(image_paths):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # Read image with all channels (RGBA if exists)
        if img.shape[2] == 3:  # If the image doesn't have an alpha channel, add one
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        memmap_file[i] = img

    # Flush changes to disk
    memmap_file.flush()

    # Return path to memmap and frame details
    return memmap_path, num_frames, frame_width, frame_height


def animate_eye(memmap_path, num_frames, frame_height, frame_width, background_image_path, enable_y_coords=False, debug=False):
    global face_x_coordinate, face_y_coordinate, latest_webcam_frame

    # Load the memmap from disk
    memmap_file = np.memmap(memmap_path, dtype='uint8', mode='r', shape=(num_frames, frame_height, frame_width, 4))

    # Load the background image
    background_img = cv2.imread(background_image_path, cv2.IMREAD_UNCHANGED)

    # Resize the background image to match the frame dimensions
    background_img = cv2.resize(background_img, (frame_width, frame_height), interpolation=cv2.INTER_AREA)

    # If the background has an alpha channel, convert it to BGR
    if background_img.shape[2] == 4:
        background_img = cv2.cvtColor(background_img, cv2.COLOR_BGRA2BGR)

    # Set initial frame index for the main loop
    frame_idx = 0

    # Variables to track the last known position
    last_move_x = 0
    last_move_y = 0

    while True:
        # Read the current frame from the memmap
        current_frame = memmap_file[frame_idx]

        # Update the frame index for looping through frames
        frame_idx = (frame_idx + 1) % num_frames

        with face_x_lock:
            if face_x_coordinate is not None:
                # Calculate the relative position of the face in the webcam frame
                relative_x_position = face_x_coordinate
                relative_y_position = face_y_coordinate if enable_y_coords and face_y_coordinate is not None else 0.5

                # Invert the movement to behave like a mirror for x and y
                move_x = int((0.5 - relative_x_position) * MOVEMENT_SCALE * frame_width)
                move_y = int((relative_y_position - 0.5) * MOVEMENT_SCALE * frame_height) if enable_y_coords else 0

                # Update the last known positions
                last_move_x = move_x
                last_move_y = move_y

            else:
                # Use the last known position if no face is detected
                move_x = last_move_x
                move_y = last_move_y

        # Create an empty canvas for the foreground frame
        composite_fg_frame = np.zeros((frame_height, frame_width, 4), dtype=np.uint8)

        # Place the current frame in the foreground frame
        composite_fg_frame[:frame_height, :frame_width] = current_frame

        # Handle overflow and fill gaps with black
        if composite_fg_frame.shape[2] == 4:  # Check if the image has an alpha channel
            # Separate the BGR and Alpha channels
            bgr, alpha = composite_fg_frame[:, :, :3], composite_fg_frame[:, :, 3]

            # Create an empty canvas (all black) for the final image
            centered_background = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255

            # Calculate valid region based on movement and clip if needed
            bg_x = move_x
            bg_y = move_y if enable_y_coords else 0

            src_x_start = max(0, -bg_x)  # Handle negative values for leftward shifts
            dst_x_start = max(0, bg_x)

            src_y_start = max(0, -bg_y) if enable_y_coords else 0
            dst_y_start = max(0, bg_y) if enable_y_coords else 0

            dst_x_end = min(frame_width, dst_x_start + (frame_width - src_x_start))
            dst_y_end = min(frame_height, dst_y_start + (frame_height - src_y_start))

            if dst_x_end > dst_x_start and dst_y_end > dst_y_start:
                centered_background[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = background_img[
                    src_y_start:src_y_start + (dst_y_end - dst_y_start),
                    src_x_start:src_x_start + (dst_x_end - dst_x_start)
                ]

            # Ensure the alpha and bgr regions match in size before blending
            min_height = min(centered_background.shape[0], bgr.shape[0])
            min_width = min(centered_background.shape[1], bgr.shape[1])

            centered_background = centered_background[:min_height, :min_width]
            bgr = bgr[:min_height, :min_width]
            alpha = alpha[:min_height, :min_width]

            # Blend the frame with the background using the alpha channel
            alpha = alpha / 255.0  # Normalize alpha to 0-1
            frame_with_bg = (alpha[..., None] * bgr + (1 - alpha[..., None]) * centered_background).astype(np.uint8)
        else:
            frame_with_bg = composite_fg_frame  # No alpha channel, just display the frame

        # Display the main animated eye frame
        cv2.imshow('Animated Eye', frame_with_bg)

        # Display the debug frame with bounding box (directly from the shared variable)
        with frame_lock:
            if debug and (latest_webcam_frame is not None):
                cv2.imshow('Debug Face Detection', latest_webcam_frame)
                # Bring the debug window to the front (Linux with wmctrl)
                os.system('wmctrl -r "Debug Face Detection" -b add,above')

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Ensure minimal delay between frames
            cv2.destroyAllWindows()
            return


if __name__ == "__main__":
    # Generate the memmap.
    # TODO: this can be pre-computed.
    memmap_path, num_frames, frame_width, frame_height = images_to_memmap(image_dir="cropped_eyes_2")

    # Start the threads for face detection
    face_detection_thread = threading.Thread(target=detect_face_position)
    face_detection_thread.start()

    # Set up fullscreen window
    cv2.namedWindow('Animated Eye', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Animated Eye', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Run the animation.
    animate_eye(memmap_path, num_frames, frame_height, frame_width,
                background_image_path="eyeball.png", enable_y_coords=True, debug=True)

    face_detection_thread.join()
