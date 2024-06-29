# VisionGuard (Object Detection App)

## Introduction
This project presents a comprehensive implementation of an object detection system utilizing machine learning techniques. The system, developed as a desktop application with a graphical user interface (GUI) using Tkinter, leverages the OpenCV library and a pre-trained deep learning model for real-time object detection. The application is designed to be user-friendly, offering functionalities such as starting and stopping the camera, taking screenshots, toggling object detection, enabling night mode, and recording video.

## Problem Description
Real-time object detection is a challenging problem in computer vision with applications in various fields such as surveillance, autonomous driving, and robotics. Developing an efficient, user-friendly, and interactive system that can perform real-time object detection on a desktop environment is the core problem this project aims to solve.

## Solution Overview
The object detection model, based on the Single Shot MultiBox Detector (SSD) with MobileNet architecture, is configured to identify various objects as defined by the COCO dataset. The application allows users to adjust the confidence threshold for detections and select the camera index for different camera sources. Additionally, the system supports night mode by inverting colors for better visibility in low-light conditions. A significant feature of the application is its ability to record video, displaying elapsed recording time and saving the output in AVI format.

## Libraries, Models, and Dataset
- **Libraries:**
  - OpenCV
  - Tkinter
  - PIL (Python Imaging Library)
  - json
  - time
  - os

- **Models:**
  - Single Shot MultiBox Detector (SSD) with MobileNet architecture

- **Dataset:**
  - COCO (Common Objects in Context) dataset for object classes

## Features
- **Start/Stop Camera:** Initiate or stop the live camera feed.
- **Take Screenshot:** Capture and save a screenshot from the live feed.
- **Toggle Detection:** Enable or disable object detection.
- **Night Mode:** Invert colors for better visibility in low-light conditions.
- **Video Recording:** Record live feed with detected objects and save as an AVI file.
- **Adjustable Confidence Threshold:** Set the minimum confidence level for object detection.
- **Select Camera Source:** Choose the camera index for different camera inputs.

 
