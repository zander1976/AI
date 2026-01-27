import cv2
import numpy as np

# Open front MacBook camera
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    raise RuntimeError("Cannot open MacBook camera! Check permissions and index.")

print("Welcome to My Test")
print("Press 'q' to quit. 'a' = previous mode, 'd' = next mode.")

# Filters / modes
modes = ["color", "gray", "edges", "invert", "gaussian", "median", "sepia", "cartoon"]
mode_index = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Prepare gray version for filters
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Dynamic thresholds for edges
    median_val = np.median(gray)
    lower = int(max(0, 0.4 * median_val))
    upper = int(min(255, 1.5 * median_val))

    # Apply selected processing
    mode = modes[mode_index]
    if mode == "gray":
        display_frame = gray
    elif mode == "edges":
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        display_frame = cv2.Canny(blurred, lower, upper)
    elif mode == "invert":
        display_frame = 255 - frame
    elif mode == "gaussian":
        display_frame = cv2.GaussianBlur(frame, (15, 15), 0)
    elif mode == "median":
        display_frame = cv2.medianBlur(frame, 9)
    elif mode == "sepia":
        kernel = np.array(
            [[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]]
        )
        display_frame = cv2.transform(frame, kernel)
        display_frame = np.clip(display_frame, 0, 255).astype(np.uint8)
    elif mode == "cartoon":
        gray_blur = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(
            gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9
        )
        color = cv2.bilateralFilter(frame, 9, 250, 250)
        display_frame = cv2.bitwise_and(color, color, mask=edges)
    else:
        display_frame = frame

    cv2.imshow("Camera Feed", display_frame)

    # Wait and capture key
    key = cv2.waitKey(10) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("a"):
        mode_index = (mode_index - 1) % len(modes)
    elif key == ord("d"):
        mode_index = (mode_index + 1) % len(modes)

cap.release()
cv2.destroyAllWindows()
