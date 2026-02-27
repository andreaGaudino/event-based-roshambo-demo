# Event based roshambo demo
Semester project developed for the ResProj course at Eurecom 

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

## How to run the code

Simply execute on the terminal:
### 
    python main.py

If you wish to quit the execution, press the 'q' key.
