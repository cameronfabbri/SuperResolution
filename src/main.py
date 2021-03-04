import video_thread_test
import libav_functions
import time

import torch
import torchvision.models.vgg

from torch.utils.data import DataLoader

from videodata import VideoDataset

# load the vgg16 model
vgg16 = torchvision.models.vgg16(pretrained=False)
# set it to a gpu model
vgg16 = vgg16.to(torch.device('cuda:0'))
optimizer = torch.optim.Adam(vgg16.parameters(), lr=1e-3)

# create our datasets to train on
dataset = VideoDataset(root_dir='data/train', train=True, cache_size=15)
data_loader = DataLoader(
    dataset,
    batch_size=9,
    shuffle=True,
    num_workers=0, 
    #prefetch_factor=4,
    #persistent_workers=True,
    pin_memory=True
)

print("we have {} chunks".format(dataset.data_decoder.get_num_chunks()))

for _ in range(0,100):
    dataset.data_decoder.chunk_superlist.pop()

print("we have {} chunks".format(dataset.data_decoder.get_num_chunks()))
dataset.data_decoder.build_chunk_superlist()
print("we have {} chunks".format(dataset.data_decoder.get_num_chunks()))