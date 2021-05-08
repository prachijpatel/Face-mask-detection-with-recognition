from cv2 import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import os
import pickle
import face_recognition


# pickel
with open('dataset_faces.dat', 'rb') as f:
    all_faces = pickle.load(f)

known_faces = list(all_faces.values())
known_names = list(all_faces.keys())
# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


model = load_model("models/mask_model.h5")

labels_dict = {0: 'with_mask', 1: 'without_mask'}
color_dict = {0: (0, 255, 0), 1: (255, 0, 0)}


cap = cv2.VideoCapture(0)  # Use camera 0

# We load the xml file
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


while True:
    _, img = cap.read()
    # img = cv2.imread("unknown_faces/IMG20200219173630.jpg")

    img = cv2.flip(img, 1, 1)
    face = classifier.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)

    # process_this_frame = not process_this_frame
    for (x, y, w, h) in face:
        # print(face)

        face_img = img[y:y+h, x:x+w]
        # face_location = face_recognition.face_locations(face_img)

        # ------------------------------------------
        cv2.imwrite('temp.jpg', face_img)

        test_image = image.load_img('temp.jpg', target_size=(150, 150, 3))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        label = model.predict_classes(test_image)[0][0]
        # print(label)
#         small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        if(label == 1):
            # recognition--------------------------

            small_frame = cv2.resize(face_img, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]

            face_encoding = face_recognition.face_encodings(rgb_small_frame)
            name = "Unknown"

        # print("face_encoding", face_encoding[0])
            if len(face_encoding) == 1:
                matches = face_recognition.compare_faces(
                    known_faces, face_encoding[0])
                face_distances = face_recognition.face_distance(
                    known_faces, face_encoding[0])
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]
            # text = f"{labels_dict[label]}\n{name}"
        # else:
        #     # text = labels_dict[label]
        cv2.rectangle(img, (x, y), (x+w, y+h), color_dict[label], 2)

        if label == 1:
            # cv2.rectangle(img, (x, y), (x, y+h), color_dict[label], -1)
            cv2.rectangle(img, (x, y+h),
                          (x+w, y+h+50), color_dict[label], cv2.FILLED)
            cv2.putText(img, name, (x, y+h+40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.rectangle(img, (x, y-50), (x+w, y), color_dict[label], -1)
        cv2.putText(img, labels_dict[label], (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # img = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)
    cv2.imshow('LIVE Capture', img)
    key = cv2.waitKey(10)
    # if Enter key is press then break out of the loop
    if key == 13:  # The Enter key
        break
# Stop video
cap.release()

# Close all started windows
cv2.destroyAllWindows()
