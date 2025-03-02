import cv2

# Open a connection to the webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Continuously capture frames from the webcam
while True:
    ret, frame = cap.read()  # ret is a boolean indicating if the frame was successfully read
    if not ret:
        print("Error: Could not read frame.")
        break

    # Display the frame in a window named 'Webcam'
    cv2.imshow('Webcam', frame)

    # Press 'q' to exit the loop and close the webcam window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()