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

from queue import Queue
from threading import Thread

# Should be okay because we're using spawn for multiproc, these will be unique per worker
getmaxframes_queue = Queue()
getframes_queue = Queue()
container = None

def dumbass_threading_getmaxframes(file_name):

    global container
    if container is None:
        print('pid:', os.getpid(), 'opening container for the first time')
        container = av.open(file_name, mode="r")
    

    #container = av.open(file_name, mode="r")
    video_stream = container.streams.video[0] # snag the first video stream

    video_stream_time_base = float(video_stream.time_base)
    video_stream_framerate = float(video_stream.base_rate)

    # duration divided by av's time base
    container_duration_seconds = container.duration / av.time_base

    #Calculate the number of frames this video file probably has
    num_frames = floor(video_stream_framerate * container_duration_seconds)
    #container.close()
    getmaxframes_queue.put(num_frames)
    return

def dumbass_threading_getframe(file_name, frame_number):

    global container
    if container is None:
        print('pid:', os.getpid(), 'opening container for the first time')
        container = av.open(file_name, mode="r")

    video_stream = container.streams.video[0] # snag the first video stream

    video_stream_time_base = float(video_stream.time_base)
    video_stream_framerate = float(video_stream.base_rate)

    #print("container: ", self.container)

    decoded_frame_generator = container.decode(video=0)
    #print("frame generator object: ", decoded_frame_generator)
    video_stream_time_base = float(video_stream.time_base)
    #print("stream time base: ", video_stream_time_base)
    video_stream_framerate = float(video_stream.base_rate)
    #print("stream base rate: ", video_stream_framerate)

    # duration divided by av's time base
    container_duration_seconds = container.duration / av.time_base
    #print("duration (seconds): ", container_duration_seconds)

    # Calculate the number of frames this video file probably has
    num_frames = floor(video_stream_framerate * container_duration_seconds)
    #print("Calculated number of frames is: ", num_frames)

    # Calculate the time offset from the frame we want to seek
    desired_frame = frame_number
    seek_time_seconds = desired_frame / video_stream_framerate
    seek_time_stream_timebase = round(seek_time_seconds / video_stream_time_base)

    #print("we're going to try to seek to video time: ", seek_time_stream_timebase)

    # Seek to our offset, if there isn't a keyframe, backtrack
    container.seek(offset=seek_time_stream_timebase, any_frame=False, backward=True, stream=video_stream)

    # grab a frame after seeking
    #print("got frame generator: ", decoded_frame_generator)
    frame = next(decoded_frame_generator)
    #print("got frame at displaytime: ", frame.dts)
    #print("this means we are", frame.dts * video_stream_time_base, "seconds into the video")

    frames_left_to_decode = floor(((seek_time_stream_timebase - frame.dts) * video_stream_time_base) * video_stream_framerate)
    #print(frames_left_to_decode, "frames left to decode")

    #print("decoding those frames")
    for _ in range(0,frames_left_to_decode):
        frame = next(decoded_frame_generator)
        #if frame.index % 10:
        #    print(".", end = "", flush=True)
    #print("") # newline

    #print("at frame time", frame.dts)
    display_time_s = frame.dts*video_stream_time_base
    #print("at display time", display_time_s, "seconds")
    #print("aka,", display_time_s*video_stream_framerate, "frames into the video")

    #print("Frame format is:", frame.format)

    reformatted_frame = frame.reformat(format=VideoFormat('rgb24'), src_colorspace="ITU709", dst_colorspace="ITU709")
    frame_rgb_numpy = reformatted_frame.to_ndarray()
    getframes_queue.put(frame_rgb_numpy)

    return

class VideoDataset(Dataset):

    def __init__(self, root_dir, file_name, train):
        self.root_dir = root_dir
        self.train = train
        self.file_name_full = os.path.join(root_dir, file_name)


    def __len__(self):
        x = Thread(target=dumbass_threading_getmaxframes, args=(self.file_name_full,))
        x.start()
        print('pid:', os.getpid(), "started getmaxframes")
        x.join()
        print('pid:', os.getpid(), "closed getmaxframes")
        self.total_frames = getmaxframes_queue.get()
        print("data from thread:", self.total_frames)
        return self.total_frames

    def __getitem__(self, idx):

        s = time.time()
        #print('pid:', os.getpid(), '- getting index[', idx, ']')

        x = Thread(target=dumbass_threading_getframe, args=(self.file_name_full,idx,))
        x.start()
        #print('pid:', os.getpid(), "started getframe")
        x.join()
        #print('pid:', os.getpid(), "closed getframe")

        extracted_frame = getframes_queue.get()

        if not getframes_queue.empty():
            print("This queue isn't empty and this shit isn't working as expected")
            print("size:", getframes_queue.qsize())
            print("queue object:", getframes_queue)
            exit()

        print("Time taken to decode frame:", round(time.time() - s, 3))

        return self.transform(extracted_frame)

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

