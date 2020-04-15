import cv2
import tensorflow as tf
import numpy as np

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# To capture video from webcam. 
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 700)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 700)

# BGR angry:red, disgust:green, fear:purple, happy:yellow, sad:blue, surprise: orange, neutral:grey
emotions_list = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
emotions_colors = {'Angry':(0,0,153),'Disgust':(0,153,0),'Fear':(153,0,76),'Happy':(0,153,153),'Sad':(153,76,0),'Surprise':(0,76,153),'Neutral':(64,64,64)}

# Load model
model = tf.keras.models.load_model('models//tl_w')


def cut_faces(image, faces_coord):
    faces = []
    
    for (x, y, w, h) in faces_coord:
        w_rm = int(0.3 * w / 2)
        faces.append(image[y: y + h, x + w_rm: x + w - w_rm])
         
    return faces

def resize(images, size=(48,48)):
    images_norm = []
    for image in images:
        if image.shape < size:
            image_norm = cv2.resize(image, size, 
                                    interpolation=cv2.INTER_AREA)
        else:
            image_norm = cv2.resize(image, size, 
                                    interpolation=cv2.INTER_CUBIC)
        images_norm.append(image_norm)

    return images_norm

def convert_data(face):
    # convert data to normalized rgb input tensor

    face_tensor = tf.convert_to_tensor(face)/255 # normalize
    rgb_tensor = tf.stack([face_tensor,face_tensor,face_tensor],-1)
    input_img = tf.reshape(rgb_tensor,[1,48,48,3])

    return input_img

def get_predictions(image, faces_coord):

    predictions = []

    faces = cut_faces(image, faces_coord)
    faces = resize(faces)

    for face in faces:
        data = {}

        input_img = convert_data(face)
        predictions.append( model.predict(input_img)[0] )

    return predictions

def display_predictions(img,predictions,faces):

    for n, (prediction, (x,y,w,h)) in enumerate(zip(predictions, faces)):

        pred = np.argmax(prediction)
        emotion = emotions_list[pred]
        colour = emotions_colors[emotion]
        text = emotion + ': ' + "%.2f"%(np.max(prediction)*100) + '%'


        # Plot rectangle
        rect = cv2.rectangle(img, (x, y), (x+w, y+h), colour, 2)
        cv2.putText(rect, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)
        cv2.putText(rect,"Face "+str(n+1),(x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
 
        # Plot corner stats
        cv2.putText(img, "Face "+str(n+1), (10 + (120*n),20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1) # Face i title
        for m, stat in enumerate(prediction): # classification stats
            cv2.putText(img, emotions_list[m], (10 + (120*n), 35 + (13*m)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, colour, 1)
            cv2.putText(img, "%.2f" %(stat*100)+"%", (70+(120*n),35+(13*m)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, colour, 1)




while True:
    # Read the frame
    _, img = cap.read()
    # Convert to img to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 
        scaleFactor = 1.1, 
        minNeighbors = 4,
        minSize = (48,48))

    predictions = get_predictions(gray, faces)

    display_predictions(img,predictions,faces)

    # Display
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cap.release()