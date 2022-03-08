# import streamlit library for streamlit module to create website
import streamlit as st

import os
import numpy as np
import time

# import datetime to convert time creation of image file to proper date format
import datetime

# import opencv(cv2) library for computer vision
import cv2

# import keras library for artificial neural networks or model trained with deep learning
from keras.models import load_model

# python module pygame import mixer module for loading and playing audio file
from pygame import mixer

# initialize the mixer module
mixer.init()

# load the alarm audio file with mixer module according to directory of audio file
sound = mixer.Sound("../Drowsiness detection/alarm.wav")

# load pre-trained model for face detection, left eye detection and right eye detection
face = cv2.CascadeClassifier(
    '../Drowsiness detection/haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier(
    '../Drowsiness detection/haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier(
    '../Drowsiness detection/haar cascade files\haarcascade_righteye_2splits.xml')

# load pre-trained deep learning model for drowsiness detection
model = load_model('../Drowsiness detection/models/cnncat2.h5')

# get current working directory of a process or current directory of file
path = os.getcwd()

# initialize varaible for score, thicc(thickness of rectangle border) and array to store the state of left eye and right eye
score = 0
thicc = 2
rpred = [99]
lpred = [99]

# %%
# title of the application
st.title('Driver Drowsiness Detection System')

# set layout for side bar
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# set sidebar content
st.sidebar.title('Driver Drowsiness Detection System')
st.sidebar.subheader('Action')

# %%
# set drop down select box for user to navigate different page of application
app_mode = st.sidebar.selectbox('Choose Action', [
                                'About App', 'Driver Drowsiness Detection System', "Captured Image of Drowsy User", ""])

# "About App" page
if app_mode == 'About App':
    st.subheader("About the Application")
    st.markdown(
        """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # introduction of application content
    st.markdown('''
                This is a **Driver Drowsiness Detection Application**.\n 
                
                The application takes in **live footage** that is captured by the camera that is connected to the device and feeds it into the system
                for the system to monitor the user and determine if the user is in a drowsy state or not.
                
                The system the of the application uses **haar cascade classifier** to detect the face, left eye and right eye of the user.
                
                The system the of the application uses **machine learning** that is trained using **Convolutional Neural Networks (CNN)** to classify and
                determine if user is drowsy or not.
                
                When the system detects that user is drowsy, an alarm will sound and a blink red rectangle will appear around the frame of the live footage.
                
                The system will also snap a picture of the user when the system detects that user is drowsy.
                
                The development of the **Driver Drowsiness Detection System** is made possible with the reference of an open-source project named 
                **"Driver Drowsiness Detection System with OpenCV & Keras"**.
                
                The link to the open-source project can be found [here](https://data-flair.training/blogs/python-project-driver-drowsiness-detection-system/).
                ''')

    # application example content
    st.subheader(
        "The following content is the example of the usage of the application and the interaction with the application:-")
    # st.image to display image from specified directory to webpage
    st.image("non_drowsy_example.png",
             "Picture of application monitor user and user is not drowsy")
    st.image("drowsy_example.png",
             "Picture of application monitor user and user is drowsy")

# %%
elif app_mode == 'Driver Drowsiness Detection System':
    st.subheader("System Application Window Frame")

    # initialize variable for fps counter use
    prevTime = 0
    fps = 0

    # initialize empty st frame for frame of cv2 showing live footage captured by camera
    stframe = st.empty()

    st.markdown(
        """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown('---')

    st.markdown(' ## Output Statistics (Live Footage Captured by Camera)')

    # instantiate a object for video capture module of OpenCV library for video capture live footage from primary camera device
    cap = cv2.VideoCapture(0)

    # instantiate a font from OpenCV library
    font = cv2.FONT_HERSHEY_TRIPLEX

    # get width of frame initiated by cv2 video capture (live footage capture by camera)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # initiate column for displaying stats of live video footage (Fps counter of video and width size)
    kpi1, kpi2 = st.beta_columns(2)

    with kpi1:
        st.markdown("**FrameRate**")
        kpi1_text = st.markdown("0")

    with kpi2:
        st.markdown("**Image Width**")
        kpi2_text = st.markdown("0")

    st.markdown("<hr/>", unsafe_allow_html=True)

    # capture live footage with camera until user goes to other page of application
    while cap.isOpened():
        # get video footage captured by camera and initialize variable to store each frame of video footage
        ret, frame = cap.read()
        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime

        # get height and width of frame of live footage captured by camera
        height, width = frame.shape[:2]
        # convert RGB/BGR colour space to grayscale for image segmentation
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # apply pre-defined model to detect face of person to create region of interest
        faces = face.detectMultiScale(
            gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
        # apply pre-defined model to detect left eye and right eye of person from region of interest generated
        left_eye = leye.detectMultiScale(gray)
        right_eye = reye.detectMultiScale(gray)
        # draw a rectangle on the frame/video footage as video footage captured by camera is displayed on application as the background of the text
        cv2.rectangle(frame, (0, height-50), (280, height),
                      (0, 0, 0), thickness=cv2.FILLED)

        # create rectangle on the area of face of person that is detected by the classifier
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 100, 100), 3)

        # detect right eye of person from the region of interest generated which is the face of user
        for (x, y, w, h) in right_eye:
            # draw rectangle around the right eye that is detected from the face of person
            cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 100, 100), 2)
            # get right eye of person
            r_eye = frame[y:y+h, x:x+w]
            # convert rgb colour space of right eye image to grayscale
            r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
            # resize right eye image to 24*24 pixels
            r_eye = cv2.resize(r_eye, (24, 24))
            # normalize the image of right eye data to be between value 0-1
            r_eye = r_eye/255
            r_eye = r_eye.reshape(24, 24, -1)
            r_eye = np.expand_dims(r_eye, axis=0)
            # predict if right eye is closed using pre-trained CNN model
            rpred = model.predict_classes(r_eye)
            break

        # detect left eye of person from the region of interest generated which is the face of user
        for (x, y, w, h) in left_eye:
            # draw rectangle around the left eye that is detected from the face of person
            cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 100, 100), 2)
            # get left eye of person
            l_eye = frame[y:y+h, x:x+w]
            # convert rgb colour space of left eye image to grayscale
            l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
            # resize right eye image to 24*24 pixels
            l_eye = cv2.resize(l_eye, (24, 24))
            # normalize the image of right eye data to be between value 0-1
            l_eye = l_eye/255
            l_eye = l_eye.reshape(24, 24, -1)
            l_eye = np.expand_dims(l_eye, axis=0)
            # predict if left eye is closed using pre-trained CNN model
            lpred = model.predict_classes(l_eye)
            break

        # if pre-trained CNN model detected and determined both left eye and right eye is closed or predicted as "0", score increases and text displayed on application window will change to "Closed"
        if(rpred[0] == 0 and lpred[0] == 0):
            if score < 30:
                score = score+1
            text = "Closed"
        # if pre-trained CNN model detected and determined both left eye and right eye is open or predicted as "1", score decreases and text displayed on application window will change to "Open"
        else:
            score = score-1
            text = "Open"
        # change score back to zero if score decreases below zero
        if(score < 0):
            score = 0
        # display "Score: " on application windows to display score of left eye and right eye to determine if user is drowsy and fall asleep
        cv2.putText(frame, text + ' Score:'+str(score),
                    (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

        # when score is over 18, determines that user is drowsy and falling asleep while driving
        if(score > 18):
            # person is feeling sleepy and the alarm audio is sounded to wake user
            try:
                sound.play()
            except:  # isplaying = False
                pass
            # increase thickness of red border that flashes around frame of captured live video footage when the system detect user falling asleep
            if(thicc < 16):
                thicc = thicc+2
            # decreases thickness of red border that flashes around frame of captured live video footage to stop border thickness from increasing
            else:
                thicc = thicc-2
                if(thicc < 2):
                    thicc = 2
            # draw rectangle border around entire frame of live video footage captured by camera if user is drowsy and fallin asleep
            cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
            # capture image of user when user system determine user is drowsy
            if(score == 19 and rpred[0] == 0 and lpred[0] == 0):
                cv2.imwrite(os.path.join(path, 'user_drowsy_image.jpg'), frame)

        # write and update video stat (fps and width) in real time as long as live footage is being captured by camera
        kpi1_text.write(
            f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)
        kpi2_text.write(
            f"<h1 style='text-align: center; color: red;'>{width}</h1>", unsafe_allow_html=True)

        # display captured live footage from camera onto website
        stframe.image(frame, channels='BGR', use_column_width=True)

    # stop capturing live video foorage from camera when user goes to another webapge
    cap.release()

# %%
elif app_mode == 'Captured Image of Drowsy User':
    st.sidebar.markdown('---')
    st.markdown(
        """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
    
    if (os.path.exists("user_drowsy_image.jpg")):
        # get creation time of image file that is captured by system from live footage captured by camera when system detects user is drowsy
        capture_time = os.path.getctime("user_drowsy_image.jpg")
        capture_time = datetime.datetime.fromtimestamp(capture_time)
        capture_time = str(capture_time)
        
        # display the captured image of drowsy user and the time of image being captured by the system
        st.subheader("Captured Image of User Detected in Drowsy State by System")
        st.image("user_drowsy_image.jpg",
                 "Image captured by system when drowsiness showed by user is detected")
        st.subheader("Time of Image Capture:-\n" + str(capture_time))
        st.markdown('---')
    #display message to tell user that no image is captured
    else:
        st.markdown("## System did not detect and capture image of user in drowsy state.")
