import cv2
import numpy as np

# Paths to the model and config file
model_file = r'D:\object detection\frozen_inference_graph.pb'
config_file = r'D:\object detection\faster_rcnn_inception_v2_coco_2018_01_28.pbtxt'

net = cv2.dnn.readNetFromTensorflow(model_file, config_file)

# COCO class labels (80 classes)
classes = ["background", "person", "bicycle", "car", "motorbike", "aeroplane", "bus",
           "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
           "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
           "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
           "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
           "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
           "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
           "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
           "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
           "toilet", "TV", "laptop", "mouse", "remote", "keyboard", "cell phone",
           "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
           "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

# Start video capture (0 for the built-in webcam)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare the frame for the network
    blob = cv2.dnn.blobFromImage(frame, 1.0/255.0, (300, 300), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)

    # Perform forward pass (inference)
    detections = net.forward()

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Filter out weak detections
            class_id = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw the bounding box and label
            label = f"{classes[class_id]}: {confidence:.2f}"
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, label, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Real-Time Object Detection", frame)

    # Press 'q' to quit the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()