# Driver Drowsiness Detection System

This is a driver drowsiness detection system that will alert the driver using audio when the system detects the driver feeling drowsy and is about to fall asleep. The system captures live footage of the driver through the connected camera device and identifies the eyes of the driver from the driver's facial feature with a machine learning model.

The system detects drowsiness in the driver by detecting the state of the eyes of the driver. If the driver closes his/her eyes for too long (around 5 seconds), then the system will alert the driver for the driver to be awake and regain focus on the road.

The model that is trained for classifying the state of the eyes (whether it is closed or opened) is a Concolutional Neutral Networks (CNN). The haar cascade model is used for detecting the face which then detects the eyes of the driver. 

This project is referenced from this [source](https://data-flair.training/blogs/python-project-driver-drowsiness-detection-system/). The model is a pre-trained model that is already trained and was download from the source website of this project.

---

There is a web application that is the graphical user interface of the driver drowsiness detection system. The web application captures live footage of the driver and shows the framerate of the footage. The web application will sound an alarm when it detects user is about to fall asleep while driving. 

The web application will take a picture of the driver as the driving is drowsy and falling asleep as detected by the web application. The web application will also show when the picture was taken.

---

**First**, create the conda environment by running the following command in the anaconda command prompt.

```
  conda create --name drowsy python=3.8.8
  conda activate drowsy
  pip install -r requirements.txt
```  

To run the web application python file, use the command below with **anaconda command prompt** under the right environment. Ensure that the command prompt is in the right directory where the web application python file or specified the directory of the web application python file. 

```
streamlit run driver_drowsiness_system_streamlit_app.py
```
