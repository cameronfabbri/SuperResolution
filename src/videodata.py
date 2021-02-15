"""

"""
import os
import time

import torch
import torchvision.transforms as transforms

from torch.utils.data import Dataset

from skimage import io

import av
from av import VideoFormat
from math import floor
from torchvision.utils import save_image
import torchvision.transforms as transforms

from threading import Thread

from video_thread_test import ThreadedDecoder

class VideoDataset(Dataset):

    def __init__(self, root_dir, file_name, train):
        self.root_dir = root_dir
        self.train = train
        self.file_name_full = os.path.join(root_dir, file_name)
        
        #just one for now
        self.data_decoder = ThreadedDecoder(self.file_name_full, 20)
        self.data_decoder.start()

    def __len__(self):
        return self.data_decoder.get_length()

    def __getitem__(self, idx):

        # we get a 0 referenced index, but frames start at 1
        idx = idx+1

        frame = self.data_decoder.get_frame(idx)

        return self.transform(frame)

    def transform(self, full_image):

        crop_func = transforms.RandomCrop(512)
        resize_func1 = transforms.Resize(128)
        resize_func2 = transforms.Resize(256)
        to_tensor = transforms.ToTensor()

        full_image = to_tensor(full_image)

        y = crop_func(full_image)
        #x = resize_func1(y)
        x = resize_func2(y)

        #x = (x / 127.5) - 1.
        #y = (y / 127.5) - 1.

        return x, y