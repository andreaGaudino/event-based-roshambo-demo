# Event-Based Roshambo Demo

This repository contains a desktop-based interactive demo of the "Rock-Paper-Scissors" game using an event camera.

## Project Overview
This project was developed as part of the **ResProj (Research Project) course at Eurecom**. 

The goal of this work is to port and adapt the original gesture recognition pipeline developed for the [Dextra Roshambo](https://github.com/SensorsINI/dextra-roshambo-python) project into a real-time desktop application. By leveraging the low latency of event-based vision, the system can anticipate human gestures and respond instantly.

### Key Modifications:
* Migrated the event processing pipeline to the modern **DV software** framework.
* Developed a custom GUI using **Python OpenCV** for real-time visualization and user interaction.
* Optimized the inference loop for desktop performance, removing external hardware dependencies (e.g., robotic arms).
## Setup


Create a Miniconda/Anaconda enviroment. If you do not have yet installed both of them: https://www.anaconda.com/download (Miniconda is suggested as it is way lighter).

Once installed: 
###
    conda create -n demo_roshambo python=3.9 
    conda activate demo_roshambo

#### More recent Python versions do not support some of the libraries we will need.

To install the libraries: 
###
    pip install -r requirements.txt

To install the library we use for the event camera:

###
    conda install -c conda-forge dv-processing -y

## Event-Camera Setup

To connect and stream data from the event-camera, you will need to install and configure the iniVation DV software.

### 1. Install DV Software
Download and install the DV software by following the instructions in the official documentation:
[DV Software Installation Guide](https://docs.inivation.com/software/dv/gui/install.html)

### 2. Configure the TCP Connection
In order to transfer event data from the DV software to this demo, you need to set up a network connection:
* Open the DV software.
* Create a **TCP connection node**.
* Assign an **IP address** and a **Port number** to this node.

### 3. Update Project Settings
Once your TCP node is configured, you need to link it to the project's code:
* Open the `utils.py` file.
* Update the file with the exact **IP address** and **Port number** you set in the DV software.

## How to run the code

Simply execute on the terminal:
### 
    python main.py

If you wish to quit the execution, press the 'q' key.
