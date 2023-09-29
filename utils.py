import glob
import random
from osgeo import gdal
import numpy as np
import os
import os
import sys
from PIL import Image
import cv2
import shutil
import torch.nn.functional as F

os.environ['PROJ_LIB'] = r'MyVenv\Lib\site-packages\pyproj\proj_dir\share\proj'



def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    return dataset


def writeTiff(im_data, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if dataset != None:
       dataset.SetGeoTransform(im_geotrans) 
       dataset.SetProjection(im_proj)
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset


#crop sample
def TifCrop(TifPath, SavePath, labelpath, labelsavepath, CropSize, RepetitionRate,label=True):
    for oo, pp, k in os.walk(TifPath):
        for a in k:
            print(a)
            dataset_img = readTif(os.path.join(oo, a))
            width = dataset_img.RasterXSize
            height = dataset_img.RasterYSize
            proj = dataset_img.GetProjection()
            geotrans = dataset_img.GetGeoTransform()
            img = cv2.cvtColor(cv2.imread(os.path.join(oo, a)), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
            if label:
                dataset_label = readTif(os.path.join(labelpath, a))
                label = dataset_label.ReadAsArray(0, 0, width, height).astype(np.uint8)
            new_name = len(os.listdir(SavePath)) + 1
            lenth = int(CropSize * (1-RepetitionRate))
            
            for i in range(int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
                for j in range(int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):

                    cropped = img[:, i*lenth:i*lenth + CropSize, j*lenth: j*lenth + CropSize]
                    geotranscrop = ()
                    geotranscrop = geotranscrop + (geotrans[0] + lenth*i * geotrans[2] + lenth * j * geotrans[1],) + (
                                   geotrans[1],) + (geotrans[2],) + (geotrans[3] + lenth * i * geotrans[5] + lenth * j * geotrans[4],) + (
                                   geotrans[4],) + (geotrans[5],)
                    
                    writeTiff(cropped, geotranscrop, proj, SavePath + "/%d.tif" % new_name)
                    if label:
                        cropped_label = label[i*lenth: i*lenth + CropSize, j*lenth: j*lenth + CropSize]
                        writeTiff(cropped_label, geotranscrop, proj, labelsavepath+"/%d.tif" % new_name)
                    new_name = new_name + 1

            for i in range(int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
                cropped = img[:,
                              int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                              (width - CropSize): width]
                geotranscrop = ()
                geotranscrop = geotranscrop + (geotrans[0] + lenth * i * geotrans[2] + (width-CropSize) * geotrans[1],) + (
                    geotrans[1],) + (geotrans[2],) + (geotrans[3] + lenth * i * geotrans[5] + (width-CropSize) * geotrans[4],) + (
                                   geotrans[4],) + (geotrans[5],)
                                   
                writeTiff(cropped, geotranscrop, proj, SavePath + "/%d.tif" % new_name)
                if label:
                    cropped_label = label[
                               int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                               (width - CropSize): width]
                    writeTiff(cropped_label,geotranscrop, proj, labelsavepath + "/%d.tif" % new_name)
                new_name = new_name + 1

            for j in range(int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):

                cropped = img[:,
                              (height - CropSize): height,
                              int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize]
                geotranscrop = ()
                geotranscrop = geotranscrop + (geotrans[0] +  (height-CropSize)* geotrans[2] + lenth * j * geotrans[1],) + (
                        geotrans[1],) + (geotrans[2],) + (geotrans[3] + (height-CropSize) * geotrans[5] + lenth * j * geotrans[4],) + (
                                       geotrans[4],) + (geotrans[5],)

                writeTiff(cropped, geotranscrop, proj, SavePath + "/%d.tif" % new_name)
                if label:
                    cropped_label = label[
                                        (height - CropSize): height,
                                        int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize]
                    writeTiff(cropped_label, geotranscrop, proj, labelsavepath + "/%d.tif" % new_name)


                new_name = new_name + 1


            cropped = img[:,
                          (height - CropSize): height,
                          (width - CropSize): width]
            geotranscrop = ()
            geotranscrop = geotranscrop + (
            geotrans[0] + (height - CropSize) * geotrans[2] + (width-CropSize) * geotrans[1],) + (
                               geotrans[1],) + (geotrans[2],) + (
                           geotrans[3] + (height - CropSize) * geotrans[5] + (width-CropSize) * geotrans[4],) + (
                               geotrans[4],) + (geotrans[5],)

            writeTiff(cropped, geotranscrop, proj, SavePath + "/%d.tif" % new_name)
            if label:
                cropped_label = label[(height - CropSize): height,(width - CropSize): width]
                writeTiff(cropped_label, geotranscrop, proj, labelsavepath + "/%d.tif" % new_name)



#restore tile image
def concate512(samplepath, imagepath,savepath,cropsize):
    for oo, pp, k in os.walk(samplepath):
        pid = 0
        for a in k:
            print(a)
            dataset_img = readTif(os.path.join(oo, a))

            width = dataset_img.RasterXSize
            height = dataset_img.RasterYSize
            proj = dataset_img.GetProjection()
            geotrans = dataset_img.GetGeoTransform()

            imagelarg = np.zeros([height, width], np.uint8)
            pixelneed = cropsize
            row = int(height/pixelneed)
            cloumn = int(width/pixelneed)
            nums = (row+1)*(cloumn+1)


            for i in range(row):
                for j in range(cloumn):
                    img = Image.open(os.path.join(imagepath,str(pid+(i*cloumn)+(j+1))+'.tif'))
                    img = np.array(img)
                    imagelarg[(i*pixelneed):((i+1)*pixelneed), (j*pixelneed):((j+1)*pixelneed)] = img
            for i in range(row):
                img = Image.open(os.path.join(imagepath,str(pid+(row*cloumn)+(i+1))+'.tif'))
                img = np.array(img)
                imagelarg[(i*pixelneed):((i+1)*pixelneed), (5056-pixelneed):5056] = img
            for j in range(cloumn):
                img = Image.open(os.path.join(imagepath, str(pid+row*(cloumn+1)+(j+1))+'.tif'))
                img = np.array(img)
                imagelarg[(height-pixelneed):height, (j*pixelneed):((j+1)*pixelneed)] = img
            img = Image.open(os.path.join(imagepath, str(pid+nums)+'.tif'))
            pid += nums
            img = np.array(img)
            imagelarg[(height-pixelneed):height, (width-pixelneed):width] = img
            writeTiff(imagelarg, geotrans, proj, os.path.join(savepath, a))


if __name__ == "__main__":

    # TifCrop(r"imgdir",
    #         r"cropimgdir", r'labeidir', r'croplabeldir', 512, 0)

    concate512(r'imgdir', r'prdictimgdir', r'concateimgdir', 512)