# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import cv2
import numpy as np
import glob
import os
import time
import imutils
from imutils.video import VideoStream
from imutils.face_utils import FaceAligner
import dlib
import argparse

# %%
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

def faceToVec(face_aligned):
    
    faceBlob = cv2.dnn.blobFromImage(face_aligned, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(faceBlob)
    vec = net.forward()

    return vec

def getFaceImageVec(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rect = detectFace(img)
    if rect is None:
        return None

    face_aligned = face_aligner.align(img, gray , dlib.rectangle(*rect))
    vec = faceToVec(face_aligned)
    return vec, rect, img

def topMatches(embeddings, vec):
    score = np.linalg.norm(embeddings - np.array(vec.flatten()), axis=1)
    imatches = np.argsort(score)
    score = score[imatches]
    return imatches, score

def render(image, rect, name, proba):
    startX, startY, endX, endY = rect
    text = "{}: {:.2f}%".format(name, proba * 100)
    y = startY - 10 if startY - 10 > 10 else startY + 10
    cv2.rectangle(image, (startX, startY), (endX, endY),
        (0, 0, 255), 2)
    cv2.putText(image, text, (startX, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    
def recognize(frame):
    result = getFaceImageVec(frame)
    if result is None:
        return None

    vec, rect, img = result
    imatches, score = topMatches(faces, vec)
    # showTopMatches(face_aligned, )
    imatches = imatches[score < threshold]
    print("matches: {}, score: {}".format(names[imatches].tolist(), score[:5].tolist()))
    if (len(imatches) == 0):
        print("unknown face")
        return None
    else:
        min_distance_label = names[imatches[0]]
        preds = clf.predict_proba(vec)[0]
        j = np.argmax(preds)
        name = le.classes_[j]
        proba = preds[j]
        print("prob: {}, name: {}, min name: {}".format(proba, name, min_distance_label))
        return name, proba, rect, img


# %%

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", default=0.65, help="confidence value")
args = vars(ap.parse_args())
# %%
detector = cv2.dnn.readNetFromCaffe("models/deploy.prototxt", "models/res10_300x300_ssd_iter_140000_fp16.caffemodel")
net = cv2.dnn.readNetFromTorch("models/openface.nn4.small2.v1.t7")
pose_predictor=dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
face_aligner = FaceAligner(pose_predictor)


# %%
faces = np.load("face_embedding.npy")
names = np.load("labels.npy")
threshold = 0.6

le = LabelEncoder()
labels = le.fit_transform(names)
clf = SVC(C=1.0, kernel="linear", probability=True)
clf.fit(faces, labels)
# test_file = "test/rose/rose-12.jpeg"
# img = imutils.resize(cv2.imread(test_file), width=600)
# result = recognize(img)

# if result is not None:
#     name, proba, rect, image = result
#     render(image, rect, name, proba)
#     cv2.imshow(name, image)
#     cv2.waitKey(0)


# %%
vs = VideoStream(src=0).start()
time.sleep(2.0)
print("Started")
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    recognition = recognize(frame)
    if recognition is not None:
        name, proba, rect, image = recognition
        if proba >= args["confidence"]:
          render(frame, rect, name, proba)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
cv2.destroyAllWindows()
vs.stop()


# %%



