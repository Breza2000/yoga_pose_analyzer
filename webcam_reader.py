import cv2

# Open the default webcam (0 indicates the first available webcam)
cap = cv2.VideoCapture(0)

# Check if the webcam was opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    print("Webcam opened successfully.")

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        # If a frame was not successfully read, break the loop
        if not ret:
            print("Error: Could not read frame.")
            break

        # Display the frame in a window named "Webcam Feed"
        cv2.imshow("Webcam Feed", frame)

        # Wait for 1 millisecond and check if 'q' key is pressed
        # If 'q' is pressed, break the loop to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and destroy all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()