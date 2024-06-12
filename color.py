import numpy as np
import cv2
from PIL import Image
from util import get_limits  # Ensure this function provides HSV limits for different colors

# Dictionary mapping color names to their corresponding BGR and HSV limits
colors = {
    "Red": [np.array([0, 120, 70]), np.array([10, 255, 255]), (0, 0, 255)],
    "Green": [np.array([36, 100, 100]), np.array([89, 255, 255]), (0, 255, 0)],
    "Blue": [np.array([94, 80, 2]), np.array([126, 255, 255]), (255, 0, 0)],
    "Yellow": [np.array([15, 150, 150]), np.array([35, 255, 255]), (0, 255, 255)],
}

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

while True:
    ret, frame = cap.read()  # Read a frame from the webcam
    if not ret:
        break

    # Convert frame to HSV color space
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Loop through each color and check for its presence in the frame
    for color_name, (lower_limit, upper_limit, bgr_color) in colors.items():
        # Create a mask to detect objects with the current color
        mask = cv2.inRange(hsv_image, lower_limit, upper_limit)

        # Convert the mask to PIL format to get bounding box
        mask_image = Image.fromarray(mask)
        bbox = mask_image.getbbox()  # Get the bounding box for detected colored areas

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            # Draw a rectangle around the detected object
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr_color, 2)

            # Display the name of the detected color above the rectangle
            text_position = (x1, y1 - 10)  # Position of the text above the rectangle
            cv2.putText(
                frame,
                color_name,
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                bgr_color,
                2,
            )

    # Display the frame with bounding boxes and text
    cv2.imshow("frame", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
