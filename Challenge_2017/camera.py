import cv2
import time

# Set up the video capture (0 for default camera)
cap = cv2.VideoCapture(0)

# Get the default frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('Linh.avi', fourcc, 20.0, (frame_width, frame_height))

start_time = time.time()
record_duration = 3 * 60  # 3 minutes in seconds

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break
    
    # Write the frame to the output file
    out.write(frame)
    
    # Display the frame
    cv2.imshow('Recording', frame)
    
    # Check if 3 minutes have passed
    if time.time() - start_time > record_duration:
        print("Recording completed")
        break
    
    # Press 'q' to stop early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Recording stopped by user")
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
