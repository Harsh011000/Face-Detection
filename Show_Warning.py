import cv2

# Warning symbol image path
warning_image_path = 'path/to/your/warning/image.png'
warning_image = cv2.imread(warning_image_path, cv2.IMREAD_UNCHANGED)

def show_img():
    # Overlay warning symbol on the frame
    alpha = 0.5  # Adjust transparency
    overlay = cv2.resize(warning_image, (frame.shape[1], frame.shape[0]))
    frame = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)