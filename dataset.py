from enum import Enum
from typing import Tuple

import PIL
import torch.utils.data
from PIL import Image
from torch import Tensor
import os
import glob
import h5py
import numpy as np
import scipy.io as sio
from DataAugment import DataAugment
from imgaug import augmenters as iaa

import numpy as np


def readString(strRef, dsFile):
    strObj = dsFile[strRef]
    str = ''.join(chr(i) for i in strObj)
    return str

def yieldNextFileName(nameDataset, dsFile):
    for nameArray in nameDataset:
        nameRef = nameArray[0]
        name = readString(nameRef, dsFile)
        yield name


class Dataset(torch.utils.data.Dataset):

    class Mode(Enum):
        TRAIN = 'train'
        TEST = 'test'

    def __init__(self, path_to_data_dir: str, mode: Mode):
        super().__init__()
        self.Augment = DataAugment()
        self.data_dir_name = ''
        if mode == Dataset.Mode.TRAIN:
            self.data_dir_name = 'train/'
        else:
            self.data_dir_name = 'test/'
        self.model = mode
        self.data_Image = (glob.glob(os.path.join(path_to_data_dir, self.data_dir_name) + '*png'))
        self.data_Image = sorted(self.data_Image, key = lambda i : int(os.path.splitext(os.path.basename(i))[0]))

        self.data_label = []

        if os.path.exists(os.path.join(path_to_data_dir, self.data_dir_name) + 'digitStruct.npy'):
            self.data_label = np.load(os.path.join(path_to_data_dir, self.data_dir_name) + 'digitStruct.npy')
        else:
            svhnMat = h5py.File(os.path.join(path_to_data_dir, self.data_dir_name) + 'digitStruct.mat', 'r')

            for index in range(len(svhnMat['digitStruct']['bbox'])):
                item = svhnMat['digitStruct']['bbox'][index].item()
                boxes = {}
                for key in ['label', 'left', 'top', 'width', 'height']:
                    attr = svhnMat[item][key]
                    values = [svhnMat[attr.value[i].item()].value[0][0]
                              for i in range(len(attr))] if len(attr) > 1 else [attr.value[0][0]]
                    boxes[key] = values
                self.data_label.append(boxes)

            np.save(os.path.join(path_to_data_dir, self.data_dir_name) + 'digitStruct.npy', self.data_label)

        print('Get data finish')




    def __len__(self) -> int:
        return len(self.data_label)

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        PILImage = PIL.Image.open(self.data_Image[index])
        PILImage = PILImage.resize((64, 64), PIL.Image.ANTIALIAS)
        Imagenp = np.asarray(PILImage)
        Num = len(self.data_label[index]['label'])
        temp = [Num]

        for i in range(5):
            if i < Num:
                tempnumber = int(self.data_label[index]['label'][i])
                if tempnumber == 10:
                    tempnumber = 0
                temp.append(tempnumber)
            else:
                temp.append(0)
        npout = np.asarray(temp)
        return Imagenp, npout

    @staticmethod
    def preprocess(image: PIL.Image.Image) -> Tensor:
        # TODO: CODE BEGIN
        raise NotImplementedError
        # TODO: CODE END
