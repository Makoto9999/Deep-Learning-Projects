# Real-time Mask-RCNN based on matterport
Refrence: https://github.com/Makoto9999/Mask_RCNN

##Installation
1.Install dependencies

  pip3 install -r requirements.txt
  
2.Clone this repository

3.Run setup from the repository root directory

  python3 setup.py install
  
4.Download pre-trained COCO weights (mask_rcnn_coco.h5) from the releases page.

5.(Optional) To train or test on MS COCO install pycocotools from one of these repos. They are forks of the original pycocotools with fixes for Python3 and Windows (the official repo doesn't seem to be active anymore).
