import cv2
import numpy as np
import pyopenpose as op

# OpenPose parameters
params = dict()
params["model_folder"] = "\\models\\pose\\body_25\\pose_deploy.prototxt"
params["net_resolution"] = "-1x368"

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Create OpenPose datum
datum = op.Datum()

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare the frame to be processed by OpenPose
    imageToProcess = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    datum.cvInputData = imageToProcess
    
    # Process the image
    opWrapper.emplaceAndPop([datum])

    # Get the pose keypoints
    pose_keypoints = datum.poseKeypoints

    # Draw the pose keypoints on the frame
    if pose_keypoints is not None:
        for person in pose_keypoints:
            for idx, point in enumerate(person):
                x, y, conf = point
                if conf > 0.1:  # Only draw keypoints with confidence > 0.1
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

    # Display the resulting frame
    cv2.imshow('OpenPose Real-time', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()