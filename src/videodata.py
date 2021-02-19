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

import libav_functions
import numpy

class VideoDataset(Dataset):

    def __init__(self, root_dir, train, cache_size):
        self.root_dir = root_dir
        self.train = train

        #just one for now
        self.data_decoder = ThreadedDecoder(root_dir, cache_size)
        #self.data_decoder.start()

    def __len__(self):
        return self.data_decoder.get_length()

    def __getitem__(self, idx):

        #print("requested index", idx)
        if self.data_decoder.active_buf is self.data_decoder.buf_1:
            print("1", end="", flush=True)
        elif self.data_decoder.active_buf is self.data_decoder.buf_2:
            print("2", end="", flush=True)
        return self.transform(self.data_decoder.active_buf[idx].copy())

    def swap(self):
        self.data_decoder.swap()

    # returns the number of epochs we can achive before we get full data coverage
    def get_epochs_per_dataset(self):
        return self.data_decoder.get_num_chunks()

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
