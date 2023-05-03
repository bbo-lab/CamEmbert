#!/usr/bin/python

import pypylon.pylon as pylon
from imageio import get_writer
import ffmpeg
import numpy as np
import subprocess as sp
import timeit
from timeit import default_timer as timer
import sys
import threading
import time
import queue
import cv2
import argparse
import signal
import os
from datetime import datetime

exit = threading.Event()
out_filename = 'output-filename.mp4'
#width = 1440
width = 1472
height = 1080

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-g', '--frametable',   type=str, help='Table with frametimes')
parser.add_argument('-o', '--output',       type=str, help = 'Video output file')
parser.add_argument('-i', '--input',        type=int, help = 'The Index of camera to use')
parser.add_argument('-m', '--master',       type=int, help = 'Configure camera to be triggered externally')
parser.add_argument('-s', '--slave',        type=int, help = 'Configure camera to be triggered externally')
parser.add_argument('-t', '--time',         type=float, help = 'Time to record')
parser.add_argument('-f', '--fps',          type=int, help = 'Framerate')
parser.add_argument('-e', '--encoder',      type=str, help = 'Encoder')
parser.add_argument('-c', '--color',        action='store_true', default=False, help='Color Vision')
parser.add_argument('-x', '--exposure',     type=int, help = 'ExposureTime', default=500)

args = parser.parse_args()

dummy = int(args.input) < 0
codec = None

#, '-vf', "'format=nv12,hwupload'"
if args.encoder is None:
    args.encoder = 'hevc_nvenc'
if args.encoder == 'hevc_nvenc':
    codec = ['-i', '-', '-an','-vcodec', 'hevc_nvenc']
if args.encoder == '264_vaapi':
    codec =  ['-hwaccel', 'vaapi' '-hwaccel_output_format', 'hevc_vaapi', '-vaapi_device', '/dev/dri/renderD128', '-i', '-', '-an', '-c:v', 'hevc_vaapi']
elif args.encoder == 'uncompressed':
    codec = ['-f','rawvideo']
elif args.encoder == 'dummy':
    codec = ['null']
#
        #'-vcodec', 'h264_amf',

command = [ "ffmpeg",
        '-y', # (optional) overwrite output file if it exists
        '-f', 'rawvideo',
        '-vcodec','rawvideo',
        '-s', str(width) + 'x' + str(height), # size of one frame
        '-pix_fmt', 'rgb24',
        '-r', '60', # frames per second
        '-rtbufsize', '2G',
        *codec,
        '-preset', 'medium',
        '-qmin', '10',
        '-qmax', '26',
        '-b:v', '50M',
        args.output]
print(command)
pipe = sp.Popen( command, stdin=sp.PIPE, stderr=sp.STDOUT,bufsize=1000,preexec_fn=os.setpgrp)


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    exit.set()

signal.signal(signal.SIGINT, signal_handler)

print(*command)
#ffmpeg -hwaccel cuda -hwaccel_output_format cuda -i input -c:v h264_nvenc -preset slow output

fps = int(args.fps)  # Hz
time_to_record = int(args.time)  # seconds
images_to_grab = (fps * time_to_record) // 3 * 3

class DummyResult:
    def GrabSucceeded(self):
        return True

    def __init__(self,BlockID, Array):
        self.Array = Array
        self.BlockID = BlockID

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def GetTimeStamp(self):
        return 0


class DummyCam:
    def __init__(self):
        self.images_to_grab =1000
        self.BlockID = 0
        
    def RetrieveResult(self,x,y):
        x = np.linspace(-5, 5, width)
        y = np.linspace(-5, 5, height)
        xx, yy = np.meshgrid(x, y)
        np.square(xx,out=xx)
        np.square(yy,out=yy)
        zz = xx + yy
        zz += self.BlockID * 0.1
        np.sin(zz,out=zz)
        zz *= 127
        zz += 127
        zz = zz.astype(np.uint8)
        Array = None
        if args.color:
            Array = np.tstack((zz,zz,zz))
        else:
            Array = zz
        tmp = DummyResult(self.BlockID, Array)
        self.BlockID = self.BlockID + 1
        return tmp

    def IsGrabbing(self):
        return self.BlockID < self.images_to_grab
    
    def StopGrabbing(self):
        pass

    def Close(self):
        pass


    def StartGrabbingMax(self,images_to_grab, strategy):
        self.images_to_grab = images_to_grab

if dummy:
    cam = DummyCam()
else:
    tlf = pylon.TlFactory.GetInstance()
    devices = tlf.EnumerateDevices()

    cam = pylon.InstantCamera(tlf.CreateDevice(devices[args.input]))
    cam.Open()
    #if args.featureload:
     #   pylon.FeaturePersistence.Load(args.featureload, cam.GetNodeMap(), True)
    print("Using device ", cam.GetDeviceInfo().GetModelName())
    cam.AcquisitionFrameRateEnable.SetValue(True)
    cam.AcquisitionFrameRate.SetValue(fps)
    cam.Gamma.SetValue(0.5)
    cam.ExposureTime.SetValue(args.exposure)
    cam.DeviceLinkThroughputLimitMode.SetValue('Off')

    if args.slave is not None:
        cam.TriggerMode.SetValue('On')
        cam.LineSelector.SetValue('Line' + str(args.slave))
        cam.LineInverter.SetValue(False)
        cam.TriggerDelay.SetValue(0)
        cam.AcquisitionFrameRateEnable.SetValue(False)
        cam.TriggerSelector.SetValue('FrameStart')
        cam.LineMode.SetValue('Input')
        cam.TriggerSource.SetValue('Line' + str(args.slave))
        cam.TriggerActivation.SetValue('FallingEdge')
    if args.master is not None:
        cam.TriggerMode.SetValue('Off')
        cam.LineSelector.SetValue('Line' + str(args.master))
        cam.LineInverter.SetValue(False)
        cam.TriggerDelay.SetValue(0)
        cam.LineMode.SetValue('Output')
        cam.LineSource.SetValue('ExposureActive')
    print('color',args.color)   
    if args.color is not None and args.color:
        cam.PixelFormat.SetValue('RGB8')        
    else:
        cam.PixelFormat.SetValue('Mono8')

    print(f"Recording {time_to_record} second video at {fps} fps")
cam.StartGrabbingMax(images_to_grab, pylon.GrabStrategy_OneByOne)

lastframe = -1
count = 0
arraybuffer = np.zeros((height,width,3),dtype=np.uint8)

time_s = timer()

q = queue.Queue()


def worker():
    frametimefile = None
    if args.frametable is not None:
        frametimefile = open(args.frametable, "w")
    count = 0
    while True:
        tmp = q.get()
        if tmp is None:
            break
        img, BlockID, CaptureTime, SystemCaptureTime = tmp
        vsize = np.min((img.shape[0:2],arraybuffer.shape[0:2]),axis=0)
        if frametimefile is not None:
            frametimefile.write(str(BlockID) + " " + str(CaptureTime) + " " + str(SystemCaptureTime) + '\n')
        if count % 200 == 0:
            gy, gx = np.gradient(img,axis=(0,1))
            np.square(gx,out=gx)
            np.square(gy,out=gy)
            gx += gy
            np.sqrt(gx,out=gx)
            print(*img.shape,"White","{:.3f}".format(np.count_nonzero(img == 255) / np.prod(img.shape)),"Black","{:.3f}".format(np.count_nonzero(img == 0) / np.prod(img.shape)),"Sharpness",np.average(gx))
            if args.color:
                cv2.imshow('Frame',cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            else:
                cv2.imshow('Frame',img)
            if cv2.waitKey(1) == ord('q'):
                break
        if args.color:
            arraybuffer[0:vsize[0],0:vsize[1]] = img[0:vsize[0],0:vsize[1]].astype(np.uint8,copy=False)
            pipe.stdin.write( arraybuffer.tobytes() )
        else:
            arraybuffer[0:vsize[0],0:vsize[1],count % 3] = img[0:vsize[0],0:vsize[1]].astype(np.uint8,copy=False)
            if count % 3 == 2:
                pipe.stdin.write( arraybuffer.tobytes() )
        count = count + 1
    print("Worker exited")

# Turn-on the worker thread.
th = threading.Thread(target=worker, daemon=True)
th.start()

try:
    while cam.IsGrabbing() and not exit.is_set():
        with cam.RetrieveResult(10000, pylon.TimeoutHandling_ThrowException) as res:
            if res.GrabSucceeded():
                img = res.Array
                q.put((img,res.BlockID,res.GetTimeStamp(),datetime.now().timestamp()))
                if q.qsize() == fps:
                    print("Warning ", q.qsize()//fps, " second of video enqueued")
                if res.BlockID != lastframe + 1:
                    print("dropped " + str(res.BlockID - lastframe - 1) +  " frame ",lastframe,"to",res.BlockID - 1)
                lastframe = res.BlockID
                print(res.BlockID, end='\r')
                count += 1
            else:
                print("Grab failed")
                # raise RuntimeError("Grab failed")
except Exception as inst:
    print(inst)
q.put(None)
time_e = timer()

print("Got", lastframe, "of", images_to_grab)
print(images_to_grab, "/", (time_e - time_s))
print("Framerate theoretical",images_to_grab / ((time_e - time_s)), "real",lastframe / ((time_e - time_s)))
print("Saving...", end=' ')
#process2.wait()
cam.StopGrabbing()
cam.Close()
print("Done")
th.join()
pipe.stdin.close()

