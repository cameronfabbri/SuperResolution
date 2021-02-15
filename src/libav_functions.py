import av
from av import VideoFormat
from math import floor

def open_video(file_name):
    """ Takes a path and opens a video file container handle """
    return av.open(file_name, mode="r")

def get_total_frames(video_container):
    """
    Takes an open container as an argument, and will use its framerate and playtime to calculate
    the number of frames total in the video file
    """

    video_stream = video_container.streams.video[0] # snag the first video stream

    video_stream_time_base = float(video_stream.time_base)
    video_stream_framerate = float(video_stream.base_rate)

    # duration divided by av's time base
    container_duration_seconds = video_container.duration / av.time_base

    #Calculate the number of frames this video file probably has
    return floor(video_stream_framerate * container_duration_seconds)

def get_video_frame(video_container, frame_number):
    """
    Takes an open video container and a target frame number, and will extract and return that frame
    in an rgb24 format ndarray
    """

    # input checks
    assert frame_number > 0, "Oops! Can't grab frame 0! Must start at 1"
    assert frame_number < get_total_frames(video_container), "Asked to get a frame out of range"

    video_stream = video_container.streams.video[0] # snag the first video stream

    video_stream_time_base = float(video_stream.time_base)
    video_stream_framerate = float(video_stream.base_rate)

    #print("container: ", self.container)

    decoded_frame_generator = video_container.decode(video=0)
    #print("frame generator object: ", decoded_frame_generator)
    video_stream_time_base = float(video_stream.time_base)
    #print("stream time base: ", video_stream_time_base)
    video_stream_framerate = float(video_stream.base_rate)
    #print("stream base rate: ", video_stream_framerate)

    # duration divided by av's time base
    container_duration_seconds = video_container.duration / av.time_base
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
    video_container.seek(offset=seek_time_stream_timebase, any_frame=False, backward=True, stream=video_stream)

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
    return reformatted_frame.to_ndarray()