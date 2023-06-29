#!/usr/bin/python

import pypylon.pylon as pylon
import ffmpeg
import numpy as np
import subprocess as sp
import timeit
from timeit import default_timer as timer
import sys
import threading
import time
import queue
import argparse
import signal
import os
from datetime import datetime

exit = threading.Event()
changed_exposure = threading.Event()

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-g', '--frametable', type=str, help='Table with frametimes')
parser.add_argument('-o', '--output', type=str, help='Video output file')
parser.add_argument('-i', '--input', type=int, help='The Index of camera to use')
parser.add_argument('-m', '--master', type=int, help='Configure camera to be triggered externally')
parser.add_argument('-s', '--slave', type=int, help='Configure camera to be triggered externally')
parser.add_argument('-t', '--time', type=float, help='Time to record')
parser.add_argument('-f', '--fps', type=int, help='Framerate')
parser.add_argument('-e', '--encoder', type=str, help='Encoder')
parser.add_argument('-sw', '--width', type=int, help='Width of recorded video', default=1472)
parser.add_argument('-sh', '--height', type=int, help='Height of recorded video', default=1080)
parser.add_argument('-c', '--color', action='store_true', default=False, help='Color Vision')
parser.add_argument('-x', '--exposure', type=int, help='ExposureTime', default=500)
parser.add_argument('-p', '--preview', action='store_true', default=False, help='Show preview')

args = parser.parse_args()

width = args.width
height = args.height

dummy = int(args.input) < 0
codec = None

# , '-vf', "'format=nv12,hwupload'"
if args.encoder is None:
    args.encoder = 'hevc_nvenc'
if args.encoder == 'hevc_nvenc':
    codec = ['-i', '-', '-an', '-vcodec', 'hevc_nvenc']
elif args.encoder == 'h264_nvenc':
    codec = ['-i', '-', '-an', '-vcodec', 'h264_nvenc']
elif args.encoder == '264_vaapi':
    codec = ['-hwaccel', 'vaapi' '-hwaccel_output_format', 'hevc_vaapi', '-vaapi_device', '/dev/dri/renderD128', '-i',
             '-', '-an', '-c:v', 'hevc_vaapi']
elif args.encoder == 'uncompressed':
    codec = ['-f', 'rawvideo']
elif args.encoder == 'libx264':
    codec = ['-i', '-', '-vcodec', 'libx264']
elif args.encoder == 'dummy':
    codec = ['null']
else:
    raise Exception("Encoder " + args.encoder + " not known")
#
# '-vcodec', 'h264_amf',

pipe = None
if args.output is not None:
    command = ["ffmpeg",
               '-y',  # (optional) overwrite output file if it exists
               '-f', 'rawvideo',
               '-vcodec', 'rawvideo',
               '-s', str(width) + 'x' + str(height),  # size of one frame
               '-pix_fmt', 'rgb24',
               '-r', '60',  # frames per second
               '-rtbufsize', '2G',
               *codec,
               '-preset', 'medium',
               '-qmin', '10',
               '-qmax', '26',
               '-b:v', '50M',
               args.output]
    print(command)
    pipe = sp.Popen(command, stdin=sp.PIPE, stderr=sp.STDOUT, bufsize=1000, preexec_fn=os.setpgrp)


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    exit.set()


signal.signal(signal.SIGINT, signal_handler)

# ffmpeg -hwaccel cuda -hwaccel_output_format cuda -i input -c:v h264_nvenc -preset slow output

time_to_record = int(args.time)  # seconds
images_to_grab = (args.fps * time_to_record) // 3 * 3


class DummyResult:
    def GrabSucceeded(self):
        return True

    def __init__(self, BlockID, Array):
        self.Array = Array
        self.BlockID = BlockID

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def GetTimeStamp(self):
        return 0


def get_capture_frame_string(frame):
    if frame is None:
        return "BlockId CaptureTime SystemCaptureTime ExposureTime"
    else:
        return str(frame.BlockID) + ' ' + str(frame.CaptureTime) + ' ' + str(frame.SystemCaptureTime) + ' ' + str(
            frame.ExposureTime)


class CapturedFrame:
    def __init__(self, img, BlockID=None, CaptureTime=None, SystemCaptureTime=None, ExposureTime=None):
        self.img = img
        self.BlockID = BlockID
        self.CaptureTime = CaptureTime
        self.SystemCaptureTime = SystemCaptureTime
        self.ExposureTime = ExposureTime


class IFloat:
    def __init__(self, Value=0):
        self.Value = 0

    def SetValue(self, Value):
        self.Value = Value

    def GetValue(self):
        return self.Value


class DummyCam:
    def __init__(self):
        self.images_to_grab = 1000
        self.BlockID = 0
        x = np.linspace(-5, 5, width)
        y = np.linspace(-5, 5, height)
        xx, yy = np.meshgrid(x, y)
        np.square(xx, out=xx)
        np.square(yy, out=yy)
        self.zz = xx + yy
        self.ExposureTime = IFloat(1)

    def RetrieveResult(self, x, y):
        zz = self.zz + self.BlockID * 0.1
        np.sin(zz, out=zz)
        exp = self.ExposureTime.GetValue() / 10000
        zz *= exp * 127
        zz = zz.astype(np.uint8)
        zz += int(exp * 127)
        Array = None
        if args.color:
            Array = np.dstack((zz, zz, zz))
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

    def StartGrabbingMax(self, images_to_grab, strategy):
        self.images_to_grab = images_to_grab


if dummy:
    cam = DummyCam()
else:
    tlf = pylon.TlFactory.GetInstance()
    devices = tlf.EnumerateDevices()

    cam = pylon.InstantCamera(tlf.CreateDevice(devices[args.input]))
    cam.Open()
    # if args.featureload:
    #   pylon.FeaturePersistence.Load(args.featureload, cam.GetNodeMap(), True)
    print("Using device ", cam.GetDeviceInfo().GetModelName())
    cam.AcquisitionFrameRateEnable.SetValue(True)
    cam.AcquisitionFrameRate.SetValue(args.fps)
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
    print('color', args.color)
    if args.color is not None and args.color:
        cam.PixelFormat.SetValue('RGB8')
    else:
        cam.PixelFormat.SetValue('Mono8')

    print(f"Recording {time_to_record} second video at {args.fps} fps")
cam.StartGrabbingMax(images_to_grab, pylon.GrabStrategy_OneByOne)

lastframe = -1
arraybuffer = np.zeros((height, width, 3), dtype=np.uint8)

time_s = timer()

q = queue.Queue()


def writer():
    frametimefile = None
    if args.frametable is not None:
        frametimefile = open(args.frametable, "w")
        frametimefile.write(get_capture_frame_string(None) + '\n')
    count = 0
    while True:
        tmp = q.get()
        if tmp is None:
            break
        img = tmp.img
        vsize = np.min((img.shape[0:2], arraybuffer.shape[0:2]), axis=0)
        if frametimefile is not None:
            frametimefile.write(get_capture_frame_string(tmp) + '\n')
        if count % 200 == 0:
            gy, gx = np.gradient(img, axis=(0, 1))
            np.square(gx, out=gx)
            np.square(gy, out=gy)
            gx += gy
            np.sqrt(gx, out=gx)
            print(*img.shape, "White", "{:.3f}".format(np.count_nonzero(img == 255) / np.prod(img.shape)), "Black",
                  "{:.3f}".format(np.count_nonzero(img == 0) / np.prod(img.shape)), "Sharpness", np.average(gx))
        if args.color:
            arraybuffer[0:vsize[0], 0:vsize[1]] = img[0:vsize[0], 0:vsize[1]].astype(np.uint8, copy=False)
            if pipe is not None:
                pipe.stdin.write(arraybuffer.tobytes())
        else:
            arraybuffer[0:vsize[0], 0:vsize[1], count % 3] = img[0:vsize[0], 0:vsize[1]].astype(np.uint8, copy=False)
            if count % 3 == 2:
                if pipe is not None:
                    pipe.stdin.write(arraybuffer.tobytes())
        count = count + 1
    print("Writer exited")


last_grapped_img = None


def grabber():
    count = 0
    lastframe = -1
    try:
        while cam.IsGrabbing() and not exit.is_set():
            with cam.RetrieveResult(10000, pylon.TimeoutHandling_ThrowException) as res:
                if res.GrabSucceeded():
                    img = res.Array
                    global last_grapped_img
                    last_grapped_img = img
                    q.put(CapturedFrame(img, BlockID=res.BlockID, CaptureTime=res.GetTimeStamp(),
                                        SystemCaptureTime=datetime.now().timestamp(), ExposureTime=args.exposure))
                    if q.qsize() == args.fps:
                        print("Warning ", q.qsize() // args.fps, " second of video enqueued")
                    if res.BlockID != lastframe + 1:
                        print("dropped " + str(res.BlockID - lastframe - 1) + " frame ", lastframe, "to",
                              res.BlockID - 1)
                    lastframe = res.BlockID
                    print(res.BlockID, end='\r')
                    count += 1
                else:
                    print("Grab failed")
                if changed_exposure.is_set():
                    changed_exposure.clear()
                    cam.ExposureTime.SetValue(args.exposure)
    except Exception as inst:
        print(inst)
    q.put(None)
    cam.StopGrabbing()
    cam.Close()
    print("Grabber exited")


grabber_thread = threading.Thread(target=grabber, daemon=True)
grabber_thread.start()

writer_thread = threading.Thread(target=writer, daemon=True)
writer_thread.start()
time_e = timer()

if args.preview:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import TextBox
    from matplotlib.widgets import Button

    fig, ax = plt.subplots()
    im = ax.imshow(np.random.randn(10, 10), vmin=0, vmax=255)
    ax_button_decrease_exposure_time = plt.axes([0.0, 0.0, 0.1, 0.05])
    ax_text_field_exposure_time = plt.axes([0.2, 0.0, 0.1, 0.05])
    ax_button_increase_exposure_time = plt.axes([0.3, 0.0, 0.1, 0.05])

    button_decrease_exposure_time =  Button(ax_button_decrease_exposure_time, label="<", color='pink', hovercolor='tomato')
    text_box_exposure_time = TextBox(ax_text_field_exposure_time, 'exp', initial=str(args.exposure))
    button_increase_exposure_time =  Button(ax_button_increase_exposure_time, label=">", color='pink', hovercolor='tomato')


    def submit(text):
        args.exposure = eval(text)
        changed_exposure.set()
        plt.draw()


    def decrease_exposure_time(event):
        args.exposure /= pow(10, 0.1)
        text_box_exposure_time.set_val("{:.2f}".format(args.exposure))


    def increase_exposure_time(event):
        args.exposure *= pow(10, 0.1)
        text_box_exposure_time.set_val("{:.2f}".format(args.exposure))


    text_box_exposure_time.on_submit(submit)
    button_decrease_exposure_time.on_clicked(decrease_exposure_time)
    button_increase_exposure_time.on_clicked(increase_exposure_time)


    plt.tight_layout()
    plt.show(block=False)
    while grabber_thread.is_alive():
        plt.gcf().canvas.draw_idle()
        plt.gcf().canvas.start_event_loop(0.5)
        if last_grapped_img is not None:
            im.set_array(last_grapped_img)

grabber_thread.join()
writer_thread.join()

if pipe is not None:
    pipe.stdin.close()

print("Got", lastframe, "of", images_to_grab)
print(images_to_grab, "/", (time_e - time_s))
print("Framerate theoretical", images_to_grab / ((time_e - time_s)), "real", lastframe / ((time_e - time_s)))
print("Done")
