import os
import io
import os.path
import shutil
import glob
import numpy as np
from PIL import Image
import json as JSON
import numpy as np
import base64
import PIL
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2

def img_arr_to_b64(img_arr):
    img_pil = PIL.Image.fromarray(img_arr)
    f = io.BytesIO()
    img_pil.save(f, format='png')
    img_bin = f.getvalue()
    if hasattr(base64, 'encodebytes'):
        img_b64 = base64.encodebytes(img_bin)
    else:
        img_b64 = base64.encodestring(img_bin)
    return img_b64

def dict_json(imageData,shapes,imagePath,fillColor=None,lineColor=None):
    '''

    :param imageData: str
    :param shapes: list
    :param imagePath: str
    :param fillColor: list
    :param lineColor: list
    :return: dict
    '''
    return {"imageData":imageData,"shapes":shapes,"fillColor":fillColor,
            'imagePath':imagePath,'lineColor':lineColor}

def dict_shapes(points,label,fill_color=None,line_color=None):
    return {'points':points,'fill_color':fill_color,'label':label,'line_color':line_color}


targetdir = "G:/Users/clai/OneDrive - University of St. Thomas/StudentResearch/WaterToxicity/FishEmbryoImages/Data/PixelLabelData/Clean all"
workdir = "G:/Users/clai/OneDrive - University of St. Thomas/StudentResearch/WaterToxicity/FishEmbryoImages/Data/Annotated imagesClean"



label = 'fishEye'

fillColor=[255,0,0,128]

lineColor=[0,255,0,128]


cate = [workdir + '/' + x for x in os.listdir(workdir) if os.path.isdir(workdir + '/' + x)]
all_folder = []

for idx, folder in enumerate(cate):
    all_folder += [folder + '/' + x for x in os.listdir(folder) if os.path.isdir(folder + '/' + x)]

json_list = []

for idx, folder in enumerate(all_folder):
    for js in glob.glob(folder + '/*.json'):
        json_list.append(js)


i = 1
for idx, folder in enumerate(all_folder):
    for im in glob.glob(folder + '/*.tif'):
        print('---reading the images:%s' % (im))
        im_filename = os.path.basename(im)[:-4]
        js_file = folder + '\\' + im_filename + '__SHAPES.json'
        # js_file2 = folder + '/' + im_filename + '__SHAPES.json'
        if js_file in json_list:
            with open(im, 'rb') as f:
                imageData = f.read()
                imageData = base64.b64encode(imageData).decode('utf-8')
            # imageData = img_arr_to_b64(np.array(Image.open(im))
            # imageData = imageData[1:]
            new_name = str(i) + '.tif'
            shutil.copy(im, targetdir+'/'+new_name)

            try:
                json_fish = JSON.load(open(js_file))
                fishEye_x = json_fish['fishEye']['shape']['x']
                fishEye_y = json_fish['fishEye']['shape']['y']

                fishEye_x_round = np.round(fishEye_x, 0)
                fishEye_y_round = np.round(fishEye_y, 0)
            except:
                print("No fishEye for: ",im)

            points = []
            shapes = []
            for j in range(fishEye_x_round.size):
                points.append([int(fishEye_x_round[j]), int(fishEye_y_round[j])])

            shapes.append(dict_shapes(points, label))
            imagePath = im
            data = dict_json(imageData, shapes, imagePath, fillColor, lineColor)

            json_file = "G:/Users/clai/OneDrive - University of St. Thomas/StudentResearch/WaterToxicity/FishEmbryoImages/Data/PixelLabelData/json/" + str(i) +'.json'
            JSON.dump(data, open(json_file, 'w'))

            i = i + 1