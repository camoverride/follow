import cv2
import os
import numpy as np
import threading
import re

# Global variables
motion_x_coordinate = None
motion_y_coordinate = None
motion_lock = threading.Lock()
latest_webcam_frame = None
frame_lock = threading.Lock()

MOVEMENT_SCALE = 0.8  # Scale factor to exaggerate the background movement


def natural_sort_key(s):
    """
    Sort filenames that contain numbers in a natural way (e.g., file1, file2, file10)
    """
    import re
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]


def detect_motion_position(debug=False, threshold=25):
    """
    Detect meaningful motion in the webcam feed and track its (x, y) position.
    """
    global motion_x_coordinate, motion_y_coordinate, latest_webcam_frame

    cap = cv2.VideoCapture(0)

    # Use the first frame as the background for comparison
    ret, first_frame = cap.read()
    if not ret:
        return

    # Convert the first frame to grayscale and blur it to reduce noise
    first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    first_frame_gray = cv2.GaussianBlur(first_frame_gray, (21, 21), 0)

    while True:
        ret, webcam_frame = cap.read()
        if not ret:
            break

        # Convert the current frame to grayscale and blur it
        current_frame_gray = cv2.cvtColor(webcam_frame, cv2.COLOR_BGR2GRAY)
        current_frame_gray = cv2.GaussianBlur(current_frame_gray, (21, 21), 0)

        # Compute the absolute difference between the current frame and the first frame
        frame_delta = cv2.absdiff(first_frame_gray, current_frame_gray)
        
        # Threshold the delta image to get the regions with motion
        _, thresh = cv2.threshold(frame_delta, threshold, 255, cv2.THRESH_BINARY)

        # Dilate the thresholded image to fill in holes and find contours
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        with motion_lock:
            # If motion is detected, find the largest contour and track its centroid
            if len(contours) > 0:
                # Find the largest contour by area
                largest_contour = max(contours, key=cv2.contourArea)

                # Ignore small movements by setting a minimum contour area threshold
                if cv2.contourArea(largest_contour) > 500:
                    # Compute the bounding box for the largest contour and track its centroid
                    (x, y, w, h) = cv2.boundingRect(largest_contour)
                    motion_x_coordinate = x + w / 2
                    motion_y_coordinate = y + h / 2

                    if debug:
                        cv2.rectangle(webcam_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                motion_x_coordinate, motion_y_coordinate = None, None

        # Update the latest frame for the main thread to access
        with frame_lock:
            latest_webcam_frame = webcam_frame.copy()

    cap.release()


def resize_images(images, new_width, new_height):
    """
    Resize the images once to the desired dimensions to reduce overhead.
    """
    resized_images = []
    for img in images:
        resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        resized_images.append(resized_img)
    return resized_images


def load_images(image_dir):
    """
    Load images into memory and return a list of images.
    """
    image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith(('png', 'jpg', 'jpeg'))]
    image_paths = sorted(image_paths, key=natural_sort_key)

    if not image_paths:
        raise ValueError("No images found in the specified directory.")

    images = []
    for img_path in image_paths:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        images.append(img)

    return images


def animate_eye(images, num_frames, frame_height, frame_width, background_image_path, enable_y_coords, debug):
    global motion_x_coordinate, motion_y_coordinate, latest_webcam_frame

    # Load the background image once and resize it to match the frame dimensions
    background_img = cv2.imread(background_image_path, cv2.IMREAD_UNCHANGED)
    background_img = cv2.resize(background_img, (frame_width, frame_height), interpolation=cv2.INTER_AREA)

    # If the background has an alpha channel, convert it to BGR
    if background_img.shape[2] == 4:
        background_img = cv2.cvtColor(background_img, cv2.COLOR_BGRA2BGR)

    frame_idx = 0
    last_move_x = 0
    last_move_y = 0

    while True:
        # Read the current frame
        current_frame = images[frame_idx]
        frame_idx = (frame_idx + 1) % num_frames

        with motion_lock:
            if motion_x_coordinate is not None:
                relative_x_position = motion_x_coordinate / frame_width
                relative_y_position = motion_y_coordinate / frame_height if enable_y_coords else 0.5

                move_x = int((0.5 - relative_x_position) * MOVEMENT_SCALE * frame_width)
                move_y = int((relative_y_position - 0.5) * MOVEMENT_SCALE * frame_height) if enable_y_coords else 0

                last_move_x = move_x
                last_move_y = move_y
            else:
                move_x = last_move_x
                move_y = last_move_y

        # Handle alpha blending only if necessary
        if current_frame.shape[2] == 4:
            bgr = current_frame[:, :, :3]
            alpha = current_frame[:, :, 3] / 255.0

            centered_background = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255
            bg_x, bg_y = move_x, move_y if enable_y_coords else 0

            src_x_start = max(0, -bg_x)
            dst_x_start = max(0, bg_x)

            src_y_start = max(0, -bg_y) if enable_y_coords else 0
            dst_y_start = max(0, bg_y)

            dst_x_end = min(frame_width, dst_x_start + (frame_width - src_x_start))
            dst_y_end = min(frame_height, dst_y_start + (frame_height - src_y_start))

            if dst_x_end > dst_x_start and dst_y_end > dst_y_start:
                centered_background[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = background_img[
                    src_y_start:src_y_start + (dst_y_end - dst_y_start),
                    src_x_start:src_x_start + (dst_x_end - dst_x_start)
                ]

            min_height = min(centered_background.shape[0], bgr.shape[0])
            min_width = min(centered_background.shape[1], bgr.shape[1])

            centered_background = centered_background[:min_height, :min_width]
            bgr = bgr[:min_height, :min_width]
            alpha = alpha[:min_height, :min_width]

            frame_with_bg = (alpha[..., None] * bgr + (1 - alpha[..., None]) * centered_background).astype(np.uint8)
        else:
            frame_with_bg = current_frame

        cv2.imshow('Animated Eye', frame_with_bg)

        with frame_lock:
            if debug and (latest_webcam_frame is not None):
                cv2.imshow('Debug Motion Detection', latest_webcam_frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            return


if __name__ == "__main__":
    # Hide the cursor
    os.system("unclutter -idle 0 &")

    # Set the display
    os.system("export DISPLAY=:0")

    # Load images and resize them to a smaller size beforehand
    original_images = load_images(image_dir="cropped_eyes_2_resized")
    resized_images = resize_images(original_images, new_width=1920, new_height=1200)  # Adjust to the desired size

    num_frames = len(resized_images)
    frame_height, frame_width = 1200, 1920  # Match the resized dimensions

    # Start the motion detection thread
    motion_detection_thread = threading.Thread(target=detect_motion_position, args=(False,))
    motion_detection_thread.start()

    # Set up the OpenCV window for fullscreen
    cv2.namedWindow('Animated Eye', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Animated Eye', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Run the animation
    animate_eye(resized_images, num_frames, frame_height, frame_width, background_image_path="eyeball.png", enable_y_coords=True, debug=False)

    motion_detection_thread.join()
