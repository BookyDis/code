import os
import sys
import cv2
from ultralytics import YOLO

# Configuration 
model_path = 'my_model.pt'
min_thresh = 0.50
cam_index = 0
imgW, imgH = 1280, 720
record = False

# Class Alias Mapping 
# Maps all parcel-like labels to 'Parcel'
label_aliases = {
    'post man': 'Post man',
    'envelope': 'Envelope',
    'cardboard box': 'Cardboard box',
    'parcel': 'Parcel'
}

# Model Validation and Load 
if not os.path.exists(model_path):
    print('ERROR: Model file not found.')
    sys.exit()

model = YOLO(model_path, task='detect')
labels = model.names

#  Camera Initialization 
cap = cv2.VideoCapture(cam_index)
cap.set(3, imgW)
cap.set(4, imgH)

# = Recorder Initialization
if record:
    recorder = cv2.VideoWriter('demo1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (imgW, imgH))

# Bounding Box Colors 
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
               (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# Inference Loop 
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print('Camera error. Exiting.')
        break

    results = model.track(frame, verbose=False)
    detections = results[0].boxes

    for i in range(len(detections)):
        xyxy_tensor = detections[i].xyxy.cpu()
        xyxy = xyxy_tensor.numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy.astype(int)

        classidx = int(detections[i].cls.item())
        raw_label = labels[classidx].strip().lower()
        conf = detections[i].conf.item()

        if conf > min_thresh:
            mapped_label = label_aliases.get(raw_label, None)

            if mapped_label:
                color = bbox_colors[classidx % 10]
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

                label = f'{mapped_label}: {int(conf * 100)}%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_ymin = max(ymin, labelSize[1] + 10)
                cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10),
                              (xmin + labelSize[0], label_ymin + baseLine - 10), color, cv2.FILLED)
                cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    if record:
        recorder.write(frame)

    cv2.imshow('Parcel Detector', frame)
    key = cv2.waitKey(5)
    if key in [ord('q'), ord('Q')]:
        break
    elif key in [ord('s'), ord('S')]:
        cv2.waitKey()
    elif key in [ord('p'), ord('P')]:
        cv2.imwrite('capture.png', frame)

# Cleanup 
cap.release()
if record:
    recorder.release()
cv2.destroyAllWindows()
