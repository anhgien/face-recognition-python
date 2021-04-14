# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import cv2
import os
import shutil
import glob
import dlib
import imutils
from imutils.face_utils import FaceAligner


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


# %%
detector = cv2.dnn.readNetFromCaffe("models/deploy.prototxt", "models/res10_300x300_ssd_iter_140000_fp16.caffemodel")
net = cv2.dnn.readNetFromTorch("models/openface.nn4.small2.v1.t7")
args = {
    "confidence": 0.5
}
pose_predictor=dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
face_aligner = FaceAligner(pose_predictor)
files = glob.glob(os.path.sep.join(["dataset","*/*"]))
output = "dataset-crop"
embeddings = []
names = []

shutil.rmtree(output)
os.mkdir(output)

for f in files:
    print("[INFO] File {}".format(f))
    img = cv2.imread(f)
    if img is None:
        continue
    img = imutils.resize(img, width=600)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rect = detectFace(img)
    if rect is None:
        continue

    face_aligned = face_aligner.align(img, gray , dlib.rectangle(*rect))
    vec = faceToVec(face_aligned)
    embeddings.append(vec.flatten())
    label = f.split(os.path.sep)[-2]
    names.append(label)

    fileName = f.split(os.path.sep)[-1]
    dirPath = os.path.sep.join([output, label])

    print("[INFO] Dir {}".format(dirPath))
    if not os.path.exists(dirPath):
        os.mkdir(dirPath)

    dest = os.path.sep.join([dirPath, fileName])
    print(["[INFO] Dest {}".format(dest)])
    cv2.imwrite(dest, face_aligned)

embeddings = np.array(embeddings)
names = np.array(names)
print("names: {}".format(names))
np.save('face_embedding.npy', embeddings)
np.save('labels.npy', names)
print("Done")


