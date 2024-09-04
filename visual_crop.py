import cv2
import os



def visual_crop(input_directory, output_directory):
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Get the list of image files from the input directory
    image_files = [f for f in os.listdir(input_directory) if f.endswith(('png', 'jpg', 'jpeg'))]

    if not image_files:
        print("No images found in the input directory.")
        return

    # Load the first image to set up the cropping window
    first_image_path = os.path.join(input_directory, image_files[0])
    first_image = cv2.imread(first_image_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel if present
    
    # Initialize the cropping box coordinates
    cropping = False
    x_start, y_start, x_end, y_end = 0, 0, 0, 0
    
    def crop_rectangle(event, x, y, flags, param):
        nonlocal x_start, y_start, x_end, y_end, cropping
        
        if event == cv2.EVENT_LBUTTONDOWN:
            x_start, y_start = x, y
            cropping = True
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if cropping:
                x_end, y_end = x, y
        
        elif event == cv2.EVENT_LBUTTONUP:
            x_end, y_end = x, y
            cropping = False
    
    # Set up the window and callback
    cv2.namedWindow("Select Crop Area")
    cv2.setMouseCallback("Select Crop Area", crop_rectangle)
    
    while True:
        img_copy = first_image.copy()
        if cropping:
            cv2.rectangle(img_copy, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
        
        cv2.imshow("Select Crop Area", img_copy)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('c') and not cropping:
            break
    
    cv2.destroyAllWindows()

    # Ensure that the coordinates are within the image bounds
    x_start, x_end = sorted([max(0, min(x_start, x_end)), min(first_image.shape[1], max(x_start, x_end))])
    y_start, y_end = sorted([max(0, min(y_start, y_end)), min(first_image.shape[0], max(y_start, y_end))])

    # Crop all images in the input directory using the selected region
    for image_file in image_files:
        img_path = os.path.join(input_directory, image_file)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel if present
        cropped_img = img[y_start:y_end, x_start:x_end]
        
        # Preserve the file extension
        output_path = os.path.join(output_directory, image_file)
        cv2.imwrite(output_path, cropped_img)

    print(f"All images have been cropped and saved to {output_directory}")


if __name__ == "__main__":
    visual_crop(input_directory="eye_no_sclera",
                output_directory="cropped_eyes")
