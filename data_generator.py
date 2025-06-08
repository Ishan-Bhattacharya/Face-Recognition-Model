import cv2
import os

# Set the output directory
output_dir = 'NAME_OF_OUTPUT_DIRECTORY'  # Set the output directory - dim_light/ daylight/ room_light/ room_light_shade

# Declare the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

capture = cv2.VideoCapture(0)

# Record the count for faces and the maximum number of faces
image_cnt = 0
max_img_cnt = 1250

while image_cnt < max_img_cnt:
    ret, frame = capture.read()  # Set the frame rate
    if not ret:
        print("Failed to capture frame!!")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert the frame to grayscale

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)) # Call the Haar cascade to detect faces

    if len(faces) == 1: # Operate when there is only one face in the frame

        x, y, w, h = faces[0] # Get the coordinates of the face

        
        margin = 15 # Set the margin so that the face is not too close to the edge

        # Get the region of interest
        # x1, y1, x2, y2
        x1 = max(0, x - margin) # x1 and y1 is the origin of the region of interest
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + w + margin) # x2 and y2 is the end of the region of interest
        y2 = min(frame.shape[0], y + h + margin)

        face_roi = frame[y1:y2, x1:x2] # Crop the region of interest

        face_roi_path = os.path.join(output_dir, f"face_{image_cnt}.jpg") # Set the output path
        cv2.imwrite(face_roi_path, face_roi) # Save the region of interest

        image_cnt += 1

        print(f"Captured face {image_cnt}/{max_img_cnt}")

    
    cv2.imshow('Face Capture', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()

print(f"Total faces captured: {image_cnt}")