import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
import src.networks as networks
import src.libav_functions

import random

checkpoint = torch.load("data/models/model.pth")
resblocks = networks.RRDB_Resblocks(5)
model = networks.Generator(resblocks)
model.load_state_dict(checkpoint['model_state'])
model.eval()


#video_file = "C:/Users/Dominic/Desktop/temp_torrents/ghost_stories_dvd_rip/Disc 1/title_t01.mkv"
video_file = "C:/Users/Dominic/Desktop/SuperResolution/test.mkv"
#video_file = "C:/Users/Dominic/Desktop/temp_torrents/Ghost Stories (Gakkou no Kaidan) (2000) [kuchikirukia]/01. Ghost Stories (2000) [DVD 480p Hi10P AC3 dual-audio][kuchikirukia].mkv"
total_frames = src.libav_functions.get_total_frames(video_file)
print("total video frames:", total_frames)
desired_frame = random.randint(0,total_frames)
print("Trying upscaling on frame number", desired_frame)
frame_numpy = src.libav_functions.get_video_frame(
        video_file,
        desired_frame
    )
frame_tensor_raw = transforms.ToTensor()(frame_numpy)
original_frame_dims = list(frame_tensor_raw.size())[-2:]
frame_tensor = frame_tensor_raw.unsqueeze(0)
result = model(frame_tensor)
result_dims = list(result.size())[-2:]
print("orig dims:", original_frame_dims)
print("result dims:", result_dims)

orig_resized = transforms.Resize(result_dims)(frame_tensor)
pad_x = result_dims[0] - original_frame_dims[0]
pad_y = result_dims[1] - original_frame_dims[1]
#orig_resized = transforms.Pad(padding=(pad_y,pad_x,0,0))(frame_tensor)
canvas = torch.cat([orig_resized[:1], result[:1]], axis=3)
save_image(canvas, "test/test_output.png")