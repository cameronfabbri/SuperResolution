import time
import random

import av
from av import VideoFormat
from math import floor
from torchvision.utils import save_image
import torchvision.transforms as transforms

container = av.open("data/train/vid1.mkv", mode="r")
video_stream = container.streams.video[0] # snag the first video stream

def GetTotalFrames():
    decoded_frame_generator = container.decode(video=0)

    video_stream_time_base = float(video_stream.time_base)
    video_stream_framerate = float(video_stream.base_rate)

    # duration divided by av's time base
    container_duration_seconds = container.duration / av.time_base

    # Calculate the number of frames this video file probably has
    num_frames = floor(video_stream_framerate * container_duration_seconds)

    return num_frames


def GetFrame(frame_number):
    print("container: ", container)

    decoded_frame_generator = container.decode(video=0)
    print("frame generator object: ", decoded_frame_generator)

    video_stream_time_base = float(video_stream.time_base)
    print("stream time base: ", video_stream_time_base)
    video_stream_framerate = float(video_stream.base_rate)
    print("stream base rate: ", video_stream_framerate)

    # duration divided by av's time base
    container_duration_seconds = container.duration / av.time_base
    print("duration (seconds): ", container_duration_seconds)

    # Calculate the number of frames this video file probably has
    num_frames = floor(video_stream_framerate * container_duration_seconds)
    print("Calculated number of frames is: ", num_frames)


    # Calculate the time offset from the frame we want to seek
    desired_frame = frame_number
    seek_time_seconds = desired_frame / video_stream_framerate
    seek_time_stream_timebase = round(seek_time_seconds / video_stream_time_base)

    print("we're going to try to seek to video time: ", seek_time_stream_timebase)

    # Seek to our offset, if there isn't a keyframe, backtrack
    container.seek(offset=seek_time_stream_timebase, any_frame=False, backward=True, stream=video_stream)

    # grab a frame after seeking
    #decoded_frame_generator = container.decode(video_stream)
    print("got frame generator: ", decoded_frame_generator)
    frame = next(decoded_frame_generator)
    print("got frame at displaytime: ", frame.dts)
    print("this means we are", frame.dts * video_stream_time_base, "seconds into the video")

    frames_left_to_decode = floor(((seek_time_stream_timebase - frame.dts) * video_stream_time_base) * video_stream_framerate)
    print(frames_left_to_decode, "frames left to decode")

    print("decoding those frames")
    for _ in range(0,frames_left_to_decode):
        frame = next(decoded_frame_generator)
        if frame.index % 10:
            print(".", end = "", flush=True)
    print("") # newline

    print("at frame time", frame.dts)
    display_time_s = frame.dts*video_stream_time_base
    print("at display time", display_time_s, "seconds")
    print("aka,", display_time_s*video_stream_framerate, "frames into the video")

    print("Frame format is:", frame.format)

    reformatted_frame = frame.reformat(format=VideoFormat('rgb24'), src_colorspace="ITU709", dst_colorspace="ITU709")
    frame_rgb_numpy = reformatted_frame.to_ndarray()
    return frame_rgb_numpy
    


total_frames = GetTotalFrames()

random.seed(time.time_ns())

num_loop = 50
total_time = 0
for i in range(0,num_loop):
    desired_frame = random.randint(1,total_frames)
    start_time = time.time()
    GetFrame(desired_frame)
    end_time = time.time()
    total_time += (end_time - start_time)

print("average time for framegrab operation is", total_time/num_loop)

frame = GetFrame(500)
save_image(transforms.ToTensor()(frame), 'test/'+ 'videoframe.png')

container.close()

container1 = av.open("data/train/vid1.mkv", mode="r")
container2 = av.open("data/train/vid1.mkv", mode="r")
container3 = av.open("data/train/vid1.mkv", mode="r")
container4 = av.open("data/train/vid1.mkv", mode="r")

print(container1)
print(container2)
print(container3)
print(container4)

container4.close()
container3.close()
container2.close()
container1.close()