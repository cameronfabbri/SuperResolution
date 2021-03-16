"""

"""
import os
import time

from math import floor
from threading import Thread

import numpy

from skimage import io

import av
import torch
import src.libav_functions
import torchvision.transforms as transforms

from av import VideoFormat
from torchvision.utils import save_image
from src.video_frame_loader import ThreadedDecoder
from torch.utils.data import Dataset


class VideoDataset(Dataset):

    def __init__(self, root_dir, train, cache_size, patch_size, num_ds):
        self.root_dir = root_dir
        self.train = train

        self.patch_size = patch_size
        self.input_size = self.patch_size // num_ds

        #just one for now
        self.data_decoder = ThreadedDecoder(root_dir, cache_size)
        #self.data_decoder.start()

    def __len__(self):
        return self.data_decoder.get_length()

    def __getitem__(self, idx):

        #print("requested index", idx)
        #if self.data_decoder.active_buf is self.data_decoder.buf_1:
        #    print("1", end="", flush=True)
        #elif self.data_decoder.active_buf is self.data_decoder.buf_2:
        #    print("2", end="", flush=True)

        # downsized and ground truth
        small, ground_truth = self.transform(self.data_decoder.active_buf[idx].copy())

        # Randomized tensor for the descrimintor training
        random_input = torch.randn((3, self.input_size, self.input_size))

        #return self.transform(self.data_decoder.active_buf[idx].copy())
        return (small, ground_truth, random_input)

    def swap(self):
        self.data_decoder.swap()

    # returns the number of epochs we can achive before we get full data
    # coverage
    def get_epochs_per_dataset(self):
        return self.data_decoder.get_num_chunks()

    def transform(self, full_image):

        crop_func = transforms.RandomCrop(self.patch_size)
        resize_func = transforms.Resize(self.input_size)

        to_tensor = transforms.ToTensor()

        full_image = to_tensor(full_image)

        if self.train:
            y = crop_func(full_image)
            # Random color jitter
            #y = transforms.ColorJitter(hue=0.3)(y)

            # Give our image a random blur between a range
            x = transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0))(y)

            x = resize_func(y)
            return x, y

        return full_image
