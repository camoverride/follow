import cv2
import os
import numpy as np
import mediapipe as mp
import threading
import random

# Global variables
face_x_coordinate = None
face_y_coordinate = None
face_x_lock = threading.Lock()
latest_webcam_frame = None
frame_lock = threading.Lock()

GRID_SIZE = 5  # Number of repetitions for grid (5x5)
MOVEMENT_SCALE = 1  # Scale factor to exaggerate the background movement

def detect_face_position():
    global face_x_coordinate, face_y_coordinate, latest_webcam_frame
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    cap = cv2.VideoCapture(0)

    while True:
        ret, webcam_frame = cap.read()
        if not ret:
            break

        # Perform face detection
        webcam_frame_rgb = cv2.cvtColor(webcam_frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(webcam_frame_rgb)

        with face_x_lock:
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    face_x_coordinate = bbox.xmin + bbox.width / 2  # Normalize face x position
                    face_y_coordinate = bbox.ymin + bbox.height / 2  # Normalize face y position

                    # Draw bounding box for debugging
                    x_min = int(bbox.xmin * webcam_frame.shape[1])
                    y_min = int(bbox.ymin * webcam_frame.shape[0])
                    width = int(bbox.width * webcam_frame.shape[1])
                    height = int(bbox.height * webcam_frame.shape[0])
                    cv2.rectangle(webcam_frame, (x_min, y_min), (x_min + width, y_min + height), (0, 255, 0), 2)

            else:
                face_x_coordinate = None  # No face detected
                face_y_coordinate = None  # No face detected

        # Update the latest frame for the main thread to access
        with frame_lock:
            latest_webcam_frame = webcam_frame.copy()

    cap.release()

def precompute_composite_frames(image_directory):
    """Precompute all composite frames with random start points for each grid cell."""
    # Get the list of image files from the directory, sorted by filename
    image_files = sorted([f for f in os.listdir(image_directory) if f.endswith(('png', 'jpg', 'jpeg'))])

    if not image_files:
        print("No images found in the input directory.")
        return []

    # Load all frames into memory
    frames = [cv2.imread(os.path.join(image_directory, image_file), cv2.IMREAD_UNCHANGED) for image_file in image_files]
    num_frames = len(frames)
    frame_height, frame_width = frames[0].shape[:2]

    # Initialize random starting points for each cell in the 5x5 grid
    random_start_points = [[random.randint(0, num_frames - 1) for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

    return frames, random_start_points, num_frames, frame_height, frame_width

def create_composite_image(image, grid_size):
    """Create a composite image by tiling the input image into a grid."""
    img_height, img_width = image.shape[:2]
    composite_image = np.tile(image, (grid_size, grid_size, 1))
    return composite_image

def animate_eye(frames, random_start_points, num_frames, frame_height, frame_width, background_image_path, enable_y_coords=False):
    global face_x_coordinate, face_y_coordinate, latest_webcam_frame

    # Load the background image
    background_img = cv2.imread(background_image_path, cv2.IMREAD_UNCHANGED)

    # Resize the background image to match the frame dimensions
    background_img = cv2.resize(background_img, (frame_width, frame_height), interpolation=cv2.INTER_AREA)

    # If the background has an alpha channel, convert it to BGR
    if background_img.shape[2] == 4:
        background_img = cv2.cvtColor(background_img, cv2.COLOR_BGRA2BGR)

    # Create tiled composite versions for the grid (5x5)
    composite_background = create_composite_image(background_img, GRID_SIZE)  # 5x5 background

    # Get composite image dimensions
    composite_bg_height, composite_bg_width = composite_background.shape[:2]
    composite_fg_height, composite_fg_width = frame_height * GRID_SIZE, frame_width * GRID_SIZE

    # Set initial frame index for the main loop
    frame_indices = [[start for start in row] for row in random_start_points]

    while True:
        # Create an empty canvas for the composite foreground frame
        composite_fg_frame = np.zeros((composite_fg_height, composite_fg_width, 4), dtype=np.uint8)

        # Fill each grid cell with the corresponding frame
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                # Get the current frame for this cell, using its independent frame index
                current_frame_idx = frame_indices[row][col]
                current_frame = frames[current_frame_idx]

                # Determine the position in the composite grid
                y_start = row * frame_height
                x_start = col * frame_width
                composite_fg_frame[y_start:y_start + frame_height, x_start:x_start + frame_width] = current_frame

                # Update the frame index for this cell, looping through the video frames
                frame_indices[row][col] = (frame_indices[row][col] + 1) % num_frames

        with face_x_lock:
            if face_x_coordinate is not None:
                # Calculate the relative position of the face in the webcam frame
                relative_x_position = face_x_coordinate
                relative_y_position = face_y_coordinate if enable_y_coords and face_y_coordinate is not None else 0.5

                # Invert the movement to behave like a mirror for x and y
                move_x = int((0.5 - relative_x_position) * MOVEMENT_SCALE * frame_width)
                move_y = int((relative_y_position - 0.5) * MOVEMENT_SCALE * frame_height) if enable_y_coords else 0

            else:
                move_x = 0  # Keep the background centered if no face is detected
                move_y = 0

        # Handle overflow and fill gaps with black
        if composite_fg_frame.shape[2] == 4:  # Check if the image has an alpha channel
            # Separate the BGR and Alpha channels
            bgr, alpha = composite_fg_frame[:, :, :3], composite_fg_frame[:, :, 3]

            # Create an empty canvas (all black) for the final image
            centered_background = np.zeros((composite_fg_height, composite_fg_width, 3), dtype=np.uint8)

            # Calculate where the background should be placed
            bg_x = move_x
            bg_y = move_y if enable_y_coords else (composite_fg_height - composite_bg_height) // 2

            # Calculate valid region based on movement and clip if needed
            src_x_start = max(0, -bg_x)  # Handle negative values for leftward shifts
            dst_x_start = max(0, bg_x)

            src_y_start = max(0, -bg_y) if enable_y_coords else 0
            dst_y_start = max(0, bg_y) if enable_y_coords else (composite_fg_height - composite_bg_height) // 2

            dst_x_end = min(composite_fg_width, dst_x_start + (composite_bg_width - src_x_start))
            dst_y_end = min(composite_fg_height, dst_y_start + (composite_bg_height - src_y_start))

            if dst_x_end > dst_x_start and dst_y_end > dst_y_start:
                centered_background[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = composite_background[
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
        cv2.imshow('Animated Eye Grid', frame_with_bg)

        # Display the debug frame with bounding box (directly from the shared variable)
        with frame_lock:
            if latest_webcam_frame is not None:
                cv2.imshow('Debug Face Detection', latest_webcam_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Ensure minimal delay between frames
            cv2.destroyAllWindows()
            return


if __name__ == "__main__":

    # Precompute the composite frames with random start points for each grid cell
    frames, random_start_points, num_frames, frame_height, frame_width = precompute_composite_frames("cropped_eyes")

    # Start the threads for face detection
    face_detection_thread = threading.Thread(target=detect_face_position)
    face_detection_thread.start()

    # Run the animation with precomputed frames, enabling y-axis tracking
    animate_eye(frames, random_start_points, num_frames, frame_height, frame_width,
                background_image_path="eyeball.png", enable_y_coords=True)

    face_detection_thread.join()
