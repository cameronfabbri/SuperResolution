import os
import sys
import queue
import time
from queue import Queue
import threading
from threading import Thread
from enum import Enum

import libav_functions

# for testing
import random
# an enum for now in case we expand commands in the future
class Command(Enum):
    STOP = 1
    ERROR = 2

class ThreadedDecoder:
    def __init__(self, file_name, max_ram_size):
        print("\n\n\n\nINIT\n\n\n\n")
        self.timeout = 30 # timeout we use for our queue operations
        self.request_queue = Queue(maxsize=1)
        self.result_queue = Queue(maxsize=1)
        self.worker_pid = None
        self.video_container = libav_functions.open_video(file_name)
        self.max_ram_size = max_ram_size

    # python has destructors... who knew? We'll use it to close our container file handle
    def __del__(self):
        # Normally we'd call self.stop() here, but upon testing, python will try to do a thread.join
        # of ALL open threads at the time... so we'll lock and wait for threads to time out by the time we get here,
        # so there's no need. Just make sure our file handler is closed
        # Close our video container so we don't leave zombie file descriptors for the OS to handle
        self.video_container.close()

    def worker_thread(self, request_queue, result_queue):
        """
        This is the processing loop we will execute within a thread.
        It will sit and wait for frame numbers or commands from the request queue,
        and place resulting grames in the result queue
        """
        # snag our pid for reporting
        thread_id = self.worker_pid.native_id
        
        # cache testing
        cache_dict = {}
        sample_entry = libav_functions.get_video_frame(self.video_container, 1)
        # turn our capacity from gigs to bytes
        max_ram_size = self.max_ram_size * 1024 * 1024 * 1024
        cache_size = max_ram_size//sample_entry.nbytes

        print("Possible cache size is:", cache_size)
        

        while True:
            # pull a request from the queue. If nothing is available for X seconds,
            # the parent is probably dead and we quit
            try:
                #print(thread_id, "About to wait on request queue")
                request = request_queue.get(block=True, timeout=self.timeout)
            except queue.Empty:
                print(thread_id, "thread timed out waiting for request. Parent died? Timeout too short?")
                return
            except:
                print(thread_id, "thread queue get errored somethign other than timeout. Here's the error:", sys.exc_info()[0])
                return

            # DEBUG print it for now
            #print(thread_id, "got request:", request)


            # If we got here we have something from the request queue
            if request == Command.STOP:
                print(thread_id, "Got stop command, ending thread.")
                return
            elif isinstance(request, int):
                print(thread_id, "thread got frame request", request)

                # check to see if it's already in our dictionary and we're not full
                if request in cache_dict:
                    print("\n\nCACHE HIT\n\n")
                    grabbed_frame = cache_dict[request]
                else:
                    try:
                        grabbed_frame = libav_functions.get_video_frame(self.video_container, request)
                    except:
                        print(thread_id, "Error grabbing frame:", sys.exc_info()[0])
                        grabbed_frame = Command.ERROR

                # if it's not in our dict and we have room
                if (request not in cache_dict) and (len(cache_dict) < cache_size):
                    # Add this frame to the dictionary
                    print("Adding frame {} to cache - [{}/{}]".format(request, len(cache_dict), cache_size))
                    cache_dict[request] = grabbed_frame
                else:
                    print("frame already in cache or no room")
                
                try:
                    self.result_queue.put(grabbed_frame, timeout=self.timeout)
                except queue.Full:
                    print(thread_id, "thread timed out trying to put result on the queue. Something isn't accepting the results")
                    return
                except:
                    print(thread_id, "Queue put errored somethign other than timeout. Here's the error:", sys.exc_info()[0])
                    return

            
    def get_length(self):
        return libav_functions.get_total_frames(self.video_container)

    def get_frame(self, frame_number):
        # make sure our thread isn't dead for some reason
        assert self.worker_pid.is_alive

        # ask our thread to decode the frame
        self.request_queue.put(frame_number)

        # wait for our result to come back
        try:
            result = self.result_queue.get(timeout=self.timeout)
        except queue.Empty:
            print("Never got a response back from the worker. Did it crash?")
            raise
        except:
            print("Queue get errored somethign other than timeout. Here's the error:", sys.exc_info()[0])
            raise
        return result

    # start the worker thread
    def start(self):
        # make sure we don't already have a thread
        assert self.worker_pid is None
        self.worker_pid = Thread(target=self.worker_thread, args=(self.request_queue, self.result_queue,))
        self.worker_pid.start()
        print("Started our worker thread")

    # send the stop command to the thread
    def stop(self):
        # make sure we actually have a thread
        assert self.worker_pid is not None

        if self.worker_pid.is_alive():
            self.request_queue.put(Command.STOP)
            print("Thread is working, sent the stop command")
        else:
            print("Thread is already dead")
    
        self.worker_pid.join()
        print("Successfully closed thread")


    #def test(self):
    #    thread = Thread(target=self.thread_test, args=(self.request_queue,))
    #    thread.start()
    #    print("started thread")
    #    thread.join()
    #    print("thread dun")
    #    return self.request_queue.get()


### test functions
# test our destructor
def test_destroy():
    testclass = ThreadedDecoder("data/train/vid1.mkv", 10)
    testclass.start()
    print("manually check logs for destructor output")
    del testclass

# check edge cases
def test_edges():
    testclass = ThreadedDecoder("data/train/vid1.mkv", 10)
    testclass.start()

    # edge cases test
    assert testclass.get_frame(0) == Command.ERROR
    print("good... shouldn't be able to get frame 0")

    assert testclass.get_frame(9999999999999) == Command.ERROR
    print("good... ridiculous get frame request should fail")

    assert testclass.get_frame(testclass.get_length() + 1) == Command.ERROR
    print("good... one outside of our range should also fail")
    testclass.stop()

def test_get_x_times(num_tries):
    testclass = ThreadedDecoder("data/train/vid1.mkv", 10)
    vidlength = testclass.get_length()
    testclass.start()
    # try num_tries frames
    for i in range(0,num_tries): 
        assert testclass.get_frame(random.randint(0, vidlength)).ndim > 0
        print("got frame {}!".format(i))
    testclass.stop()

def test_multiple_workers(num_workers, num_tries):
    readers = [ThreadedDecoder("data/train/vid1.mkv", 10) for i in range(num_workers)]
    print("starting all reader threads")
    for reader in readers:
        reader.start()
    print("telling all our readers to get some frames")
    for _ in range(0,num_tries):
        for reader in readers:
            frame = reader.get_frame(random.randint(0, reader.get_length()))
            assert frame.ndim > 0
    print("stopping our readers")
    for reader in readers:
        reader.stop()
    
def test_cache():
    testclass = ThreadedDecoder("data/train/vid1.mkv", 10)
    testclass.start()
    testclass.get_frame(5)
    testclass.get_frame(5)
    testclass.stop()

if __name__ == '__main__':
    #test_edges()
    #test_get_x_times(50)
    #test_multiple_workers(2, 10)
    #test_multiple_workers(5, 10)
    test_cache()
    #test_destroy()

    

    print("All tests passed!!!")