from ultralytics import YOLO
import cv2
import random
from sklearn.cluster import KMeans
from collections import Counter
import numpy as np


model = YOLO('C:\\Users\\corle\\runs\\detect\\the_model\\weights\\best.pt')

class_colors = {}
for cls_id in range(len(model.names)):
    class_colors[cls_id] = [random.randint(0, 255) for _ in range(3)]

cap = cv2.VideoCapture('C:\\Users\\corle\\Downloads\\v4.mp4')  # <- video yolunu buraya yaz

current_x, current_y = 400, 300  # Başlangıçta ekranın ortası

smoothing = 0.1

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, conf=0.2, verbose=False)
    x_list, y_list = [], []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls = int(box.cls[0])
            conf = box.conf[0]
            label = f'{model.names[cls]} {conf:.2f}'
            color = class_colors[cls]

            
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            x_list.append(center_x)
            y_list.append(center_y)

    
    if len(x_list) >= 2:
        points = np.array(list(zip(x_list, y_list)))
        dynamic_k = max(1, int(np.log2(len(points))))
        kmeans = KMeans(n_clusters=dynamic_k, n_init=10)
        kmeans.fit(points)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        counts = Counter(labels)
        target_cluster = counts.most_common(1)[0][0]
        target_x, target_y = centroids[target_cluster]
    elif x_list and y_list:
        target_x = sum(x_list) / len(x_list)
        target_y = sum(y_list) / len(y_list)
    else:
        target_x, target_y = current_x, current_y

    
    current_x += smoothing * (target_x - current_x)
    current_y += smoothing * (target_y - current_y)

    int_x, int_y = int(current_x), int(current_y)

    
    cv2.line(frame, (0, int_y), (frame.shape[1], int_y), (0, 0, 0), 3)
    cv2.line(frame, (int_x, 0), (int_x, frame.shape[0]), (0, 0, 0), 3)
    cv2.circle(frame, (int_x, int_y), 6, (0, 0, 255), -1)
    cv2.circle(frame, (int_x, int_y), 12, (0, 0, 0), 2)
    cv2.circle(frame, (int_x, int_y), 40, (0, 0, 0), 2)

    
    cv2.namedWindow("Tracking", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

