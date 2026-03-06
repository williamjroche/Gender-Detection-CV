import cv2
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from keras.models import load_model
import numpy as np

img_cam = cv2.VideoCapture(0)
window_name = 'Gender Detection'

#Model has 95% accuracy - change this file path to your path
gender_model = load_model('E:/personal/computer vision/custom models/gender_detection_model.keras')

#Built in OpenCV face detection --> used to capture face image for my gender det. model - change this file path to your path
face_cascade = cv2.CascadeClassifier("E:\personal\computer vision\haarcascade_frontalface_alt.xml")

#return either male or female prediction with confidence
def male_or_female(prediction):
      if prediction[0][0] > prediction[0][1]:
            return f"Female: {prediction[0][0]:.4}"
      else:
            return f"Male: {prediction[0][1]:.4}"

while True:
    success, img = img_cam.read() #read videocapture
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #built in face detection needs a grayscale image
    faces = face_cascade.detectMultiScale(  #setup/config
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )
    
    for (x,y,w,h) in faces: #gather face dimensions and position
        face_img = img[y:y+h, x:x+w]

        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB) #my gender det. model needs an RGB image

        img_height, img_width = 128, 128
        img_new = tf.image.resize(face_rgb, (img_width, img_height)) #resize to 128,128 (my model needs it to be 128,128)
        img_array = image.img_to_array(img_new) #convert to an array
        img_array = np.expand_dims(img_array, axis=0)
        pred = gender_model.predict(img_array, verbose=0)

        prediction_dsiplay = male_or_female(pred) #convert out prediciton array to a string
        print(prediction_dsiplay)
        cv2.rectangle(img, (x,y-10), (x+w,y+h+10), (0,0,0), 2) #draw rectangle around face dimensions
        cv2.putText(img, prediction_dsiplay, (x,y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2) #put text for out prediction and confidence

    cv2.imshow(window_name, img)
    if cv2.waitKey(1) & 0xFF == ord('q'): #q key to exit
            break
