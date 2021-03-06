import face_recognition
from cv2 import cv2
import numpy as np
import urllib.request
import os
import pickle
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

'''
KNOWN_FACES_DIR = 'known_faces'
print('Loading known faces...')
known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):

    # Next we load every file of faces of known person
    for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):

        # Load an image
        image = face_recognition.load_image_file(
            f'{KNOWN_FACES_DIR}/{name}/{filename}')
        encoding = face_recognition.face_encodings(image)[0]

        # Append encodings and name
        known_faces.append(encoding)
        known_names.append(name)

# print(known_faces)
# print(known_names)
'''

# opening encodings
with open('dataset_faces.dat', 'rb') as f:
    all_faces = pickle.load(f)

known_faces = list(all_faces.values())
known_names = list(all_faces.keys())
print(known_faces)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


video_capture = cv2.VideoCapture(0)

model = load_model("models/mymodel.h5")

labels_dict = {0: 'with_mask', 1: 'without_mask'}
color_dict = {0: (0, 255, 0), 1: (255, 0, 0)}
# We load the xml file
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


while True:

    frame = cv2.imread("unknown_faces/IMG20200219173630.jpg")

    # ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1)

 # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

  # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

   # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(
                known_faces, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(
                known_faces, face_encoding)
            print("face_encoding", face_encoding)

            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame
    print(face_locations)
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size

        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        x = left
        y = top
        w = right
        h = bottom
        face_img = frame[y:y+h, x:x+w]
        cv2.imwrite('temp.jpg', face_img)
        test_image = image.load_img('temp.jpg', target_size=(150, 150, 3))
        # test_image = np.reshape(frame, (1, 150, 150, 3))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        label = model.predict_classes(test_image)[0][0]

        if(label == 1):
            text = labels_dict[label] + " " + name
        else:
            text = labels_dict[label]
        # cv2.rectangle(frame, (x, y), (x+w, y+h), color_dict[label], 2)
        # cv2.rectangle(frame, (x, y-40), (x+w, y), color_dict[label], -1)
        # cv2.putText(frame, text, (x, y-10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom),
                      color_dict[label], 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 50),
                      (right, bottom), color_dict[label], cv2.FILLED)
        # font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, text, (left + 10, bottom - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

    # label = model.predict_classes(test_image)[0][0]
    # Display the resulting image
# frame = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3)
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
# video_capture.release()
cv2.destroyAllWindows()
