# Real-time Mask-RCNN based on matterport
Refrence: https://github.com/Makoto9999/Mask_RCNN

## Installation

#### 1.Create a conda virtual environment
      conda create -n MaskRCNN python=3.7 pip

#### 2.Install dependencies
      activate MaskRCNN
      pip3 install -r requirements.txt
      python3 -m pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.11.0-py3-none-any.whl
      update tensorflow
  
#### 3.Clone matterport Mask-RCNN repository
      git clone https://github.com/matterport/Mask_RCNN.git

#### 4.Run setup from the repository root directory
      python3 setup.py install
      
#### 5.Install pycocotools

##### 5.1 colone this repo
      git clone https://github.com/philferriere/cocoapi.git
      
##### 5.2 use pip to install pycocotools
      pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
      
##### 5.3 if it shows error about requirement of Visual C++, install Microsoft Build Tools 2017
      https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2017
      
      
#### 6.Download pre-trained COCO weights (mask_rcnn_coco.h5) from the releases page.
      https://github.com/matterport/Mask_RCNN/releases
      mask_rcnn_coco.h5
      


