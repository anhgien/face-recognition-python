import cv2
import numpy as np
import glob
import os
import time
import imutils
from imutils.video import VideoStream
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-l", "--label", required=True,
	help="path to dataset/<label> dir")
ap.add_argument("-c", "--confidence", default=0.6, help="confidence value")

args = vars(ap.parse_args())

def detectFace(face):
    (h, w) = face.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(face, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
    detector.setInput(blob)
    detections = detector.forward()
    i = np.argmax(detections[0, 0, :, 2])

    confidence = detections[0, 0, i, 2]
    if confidence < args["confidence"]:
        return None

    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")

    (fH, fW) = [endY - startY, endX - startX]

    if fH < 20 or fW < 20:
        return None

    return [startX, startY, endX, endY]

detector = cv2.dnn.readNetFromCaffe("models/deploy.prototxt", "models/res10_300x300_ssd_iter_140000_fp16.caffemodel")
vs = VideoStream(src=0).start()
time.sleep(2.0)
label = args["label"]
dest = os.path.sep.join(["dataset", label])
if not os.path.exists(dest):
  os.mkdir(dest)

total = 1

while True:
  frame = vs.read()
  image = imutils.resize(frame, width=600)
  rect = detectFace(image)

  if rect is not None:
    f = os.path.sep.join([dest, "{}.png".format(total)])
    print("save image {}".format(f))
    cv2.imwrite(f, frame)
    total += 1
    time.sleep(1.0)

  cv2.imshow("Frame", frame)
  key = cv2.waitKey(1) & 0xFF
  # if the `q` key was pressed, break from the loop
  if key == ord("q"):
      break
