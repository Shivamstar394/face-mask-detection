import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('mask_detector.h5')

cap = cv2.VideoCapture(0)  # 0 for default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (150, 150))
    img = np.expand_dims(img, axis=0) / 255.0

    pred = model.predict(img)[0][0]
    label = "Mask" if pred < 0.5 else "No Mask"
    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

    cv2.putText(frame, label, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow('Face Mask Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
