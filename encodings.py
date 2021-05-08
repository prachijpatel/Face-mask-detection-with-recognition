import face_recognition
import urllib.request
import os
import pickle

KNOWN_FACES_DIR = 'known_faces'

all_faces = {}
print('Loading known faces...')
known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):

    # Next we load every file of faces of known person

    for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):
        st = f'{KNOWN_FACES_DIR}/{name}'
        # Load an image
        image = face_recognition.load_image_file(
            f'{KNOWN_FACES_DIR}/{name}/{filename}')
        print(st)
        encoding = face_recognition.face_encodings(image)[0]

        # Append encodings and name
        known_faces.append(encoding)
        known_names.append(name)
        all_faces[name] = encoding
with open('dataset_faces.dat', 'wb') as f:
    pickle.dump(all_faces, f)
