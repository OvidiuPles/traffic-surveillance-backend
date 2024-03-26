import os

import cv2
import numpy as np

from ultralytics import YOLO

from source.variables import colors


def launcher():
    # training
    # model = YOLO("yolov8n.yaml")
    # model.train(data=r"source/config.yaml", epochs=1)

    # #  prediction webcam
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        H, W, _ = frame.shape

        threshold = 0.5
        results = model(frame)[0]

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > threshold:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), colors[0], 4)
                cv2.putText(frame, results.names[int(class_id)].upper() + " ID:", (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, colors[0], 3, cv2.LINE_AA)

        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # cap = cv2.VideoCapture(0)
    # while True:
    #     ret, frame = cap.read()
    #     # H, W, _ = frame.shape
    #     #
    #     # threshold = 0.5
    #     # results = model(frame)[0]
    #     #
    #     # for result in results.boxes.data.tolist():
    #     #     x1, y1, x2, y2, score, class_id = result
    #     #     if score > threshold:
    #     #         cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), colors[0], 4)
    #     #         cv2.putText(frame, results.names[int(class_id)].upper() + " ID:", (int(x1), int(y1 - 10)),
    #     #                     cv2.FONT_HERSHEY_SIMPLEX, 1.3, colors[0], 3, cv2.LINE_AA)
    #     cv2.imshow("x", frame)
    #
    # cap.release()
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    launcher()
