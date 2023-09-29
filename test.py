from osgeo import gdal
import numpy as np
import datetime
import os
import sys
import time
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import model
os.environ['PROJ_LIB'] = r'myvenv\Lib\site-packages\pyproj\proj_dir\share\proj'


def imgread(fileName):
    datagd = gdal.Open(fileName)
    width = datagd.RasterXSize
    height = datagd.RasterYSize
    data = datagd.ReadAsArray(0, 0, width, height)
    if (len(data.shape) == 3):
        data = data.swapaxes(1, 0).swapaxes(1, 2)
    return data, datagd


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
    if(dataset!= None):
        dataset.SetGeoTransform(im_geotrans) #写入仿射变换参数
        dataset.SetProjection(im_proj) #写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i+1).WriteArray(im_data[i])
    del dataset


if __name__ == '__main__':
    from glob import glob
    ModelPath = r"..\weights.pth"
    num_classes = 2
    tifimage = glob(r'MyDataset\*.tif')
    ResultPath = "resultdir"

    testtime = []
    starttime = datetime.datetime.now()
    model = trans_deeplab.DeepLab(num_classes=2, backbone="mobilenet", downsample_factor=32, pretrained=True).cuda()
    model.load_state_dict(torch.load(ModelPath))
    res = 0
    results = []
    model.eval()

    with tqdm(total=len(tifimage), desc='test', postfix=dict, mininterval=0.3, colour='cyan') as pbar:
        for i, sampled in enumerate(tifimage):
            image_batch, image_batch_gd = imgread(sampled)
            image_batch = T.ToTensor()(image_batch).cuda().unsqueeze(0)
            start = time.time()
            outputs = model(image_batch)
            end = time.time()
            res += (end - start)
            temp_inputs = torch.softmax(outputs.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1, num_classes), -1)
            result = torch.argmax(temp_inputs, -1).view(512, 512).cpu().numpy()
            writeTiff(result, image_batch_gd.GetGeoTransform(), image_batch_gd.GetProjection(), os.path.join(ResultPath,os.path.basename(sampled)))
            pbar.update(1)

    print(f'FPS is {res/(i+1)}')
    endtime = datetime.datetime.now()
    text = "模型预测完毕,目前耗时间: " + str((endtime - starttime).seconds) + "s"
    testtime.append(text)