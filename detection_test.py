from cv2 import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
model = load_model("models/mask_model.h5")

labels_dict = {0: 'with_mask', 1: 'without_mask'}
color_dict = {0: (0, 255, 0), 1: (255, 0, 0)}

cap = cv2.VideoCapture(0)  # Use camera 0

# We load the xml file
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
i = 0
while True:
    _, img = cap.read()
    # img = cv2.imread("face_reco/unknown_faces/IMG20200219173630.jpg")

    img = cv2.flip(img, 1, 1)
    face = classifier.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)
    for(x, y, w, h) in face:
        face_img = img[y:y+h, x:x+w]
        cv2.imwrite('temp.jpg', face_img)

        test_image = image.load_img('temp.jpg', target_size=(150, 150, 3))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        label = model.predict_classes(test_image)[0][0]

        # print(result)
        # for r in result:
        #     print(r)
        #     if r[0] > r[1]:
        #         label = 0
        #     else:
        #         label = 1
        # label=np.argmax(result,axis=1)[0]
        # print(label)
        cv2.rectangle(img, (x, y), (x+w, y+h), color_dict[label], 2)
        cv2.rectangle(img, (x, y-40), (x+w, y), color_dict[label], -1)
        cv2.putText(img, labels_dict[label], (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        i += 1
    # Show the image
    cv2.imshow('LIVE Capture', img)
    key = cv2.waitKey(10)
    # if Enter key is press then break out of the loop
    if key == 13:  # The Enter key
        break
    print(i)
# Stop video
cap.release()

# Close all started windows
cv2.destroyAllWindows()
