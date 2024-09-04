import cv2
import mediapipe as mp
import numpy as np
import os


def delete_sclera_from_video(path_to_eye_video, output_directory):
    # Debugging flag
    DEBUG_LANDMARKS = False

    # Load the video file
    cap = cv2.VideoCapture(path_to_eye_video)

    # Initialize MediaPipe Face Mesh with the "full" model and increased confidence
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,  # Assuming you're focusing on a single face
        refine_landmarks=True,  # Enable this to get more refined landmarks
        min_detection_confidence=0.7,  # Increase for more reliable detections
        min_tracking_confidence=0.7  # Increase for more reliable tracking
    )

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create a directory to save individual frames
    os.makedirs(output_directory, exist_ok=True)

    # Define the codec and create VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = os.path.join(output_directory, "edited_eye.mp4")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Function to create a smooth contour for the eye shape
    def create_eye_contour(eye_points):
        contour = np.array(eye_points, dtype=np.int32)
        return contour

    # Function to remove the eye region (replace with transparency) with feathered edges
    def remove_eye_region(frame, landmarks, width, height, feather_radius=4):  # Increased feather radius
        left_eye_indices = [173, 157, 158, 159, 160, 161, 246, 163, 144, 145, 153, 154, 155]
        right_eye_indices = [398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380]

        left_eye_points = [(int(landmarks[i].x * width), int(landmarks[i].y * height)) for i in left_eye_indices]
        right_eye_points = [(int(landmarks[i].x * width), int(landmarks[i].y * height)) for i in right_eye_indices]

        # Create mask with eye regions filled in
        mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        left_eye_contour = create_eye_contour(left_eye_points)
        right_eye_contour = create_eye_contour(right_eye_points)

        cv2.fillPoly(mask, [left_eye_contour], 255)
        cv2.fillPoly(mask, [right_eye_contour], 255)

        # Blur the mask to create a feathered edge with a larger radius
        feathered_mask = cv2.GaussianBlur(mask, (2 * feather_radius + 1, 2 * feather_radius + 1), feather_radius)

        # Create an alpha channel based on the feathered mask
        frame_with_alpha = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        frame_with_alpha[:, :, 3] = cv2.bitwise_not(feathered_mask)

        if DEBUG_LANDMARKS:
            for point in left_eye_points:
                cv2.circle(frame_with_alpha, point, 2, (0, 255, 0, 255), -1)
            for point in right_eye_points:
                cv2.circle(frame_with_alpha, point, 2, (0, 255, 0, 255), -1)

        return frame_with_alpha

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                frame_with_alpha = remove_eye_region(frame, face_landmarks.landmark, width, height)

        out.write(cv2.cvtColor(frame_with_alpha, cv2.COLOR_BGRA2BGR))  # Convert back to BGR for saving the video

        frame_filename = os.path.join(output_directory, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_filename, frame_with_alpha)

        cv2.imshow('Edited Eye Video', frame_with_alpha)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    delete_sclera_from_video(path_to_eye_video="cam_face.mp4",
                             output_directory="eye_no_sclera")
