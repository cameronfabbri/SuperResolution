import video_thread_test
import libav_functions
import time

data_decoder = video_thread_test.ThreadedDecoder('data/train', 10)



#data_decoder.swap()

#data_decoder.swap()

#data_decoder.swap()

#data_decoder.swap()




#testbuffer = []
#libav_functions.get_video_frames('data/train/vid1.mkv', 
#                                  start_frame=500, 
#                                  number_of_frames=500, 
#                                  target_buffer=testbuffer)

#print("buf len:", len(testbuffer))