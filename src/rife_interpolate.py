import os
import sys
rife_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../ECCV2022-RIFE')
sys.path.insert(0, rife_dir)
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings
import _thread
import skvideo.io
from queue import Queue, Empty
import sys; print(sys.path)
import model as the_model
import importlib
importlib.reload(the_model)
print(the_model.__file__)
from model.pytorch_msssim import ssim_matlab

warnings.filterwarnings("ignore")

def transferAudio(sourceVideo, targetVideo):
    import shutil
    import moviepy.editor
    tempAudioFileName = "./temp/audio.mkv"

    # split audio from original video file and store in "temp" directory
    if True:

        # clear old "temp" directory if it exits
        if os.path.isdir("temp"):
            # remove temp directory
            shutil.rmtree("temp")
        # create new "temp" directory
        os.makedirs("temp")
        # extract audio from video
        os.system('ffmpeg -y -i "{}" -c:a copy -vn {}'.format(sourceVideo, tempAudioFileName))

    targetNoAudio = os.path.splitext(targetVideo)[0] + "_noaudio" + os.path.splitext(targetVideo)[1]
    os.rename(targetVideo, targetNoAudio)
    # combine audio file and new video file
    os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(targetNoAudio, tempAudioFileName, targetVideo))

    if os.path.getsize(targetVideo) == 0: # if ffmpeg failed to merge the video and audio together try converting the audio to aac
        tempAudioFileName = "./temp/audio.m4a"
        os.system('ffmpeg -y -i "{}" -c:a aac -b:a 160k -vn {}'.format(sourceVideo, tempAudioFileName))
        os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(targetNoAudio, tempAudioFileName, targetVideo))
        if (os.path.getsize(targetVideo) == 0): # if aac is not supported by selected format
            os.rename(targetNoAudio, targetVideo)
            print("Audio transfer failed. Interpolated video will have no audio")
        else:
            print("Lossless audio transfer failed. Audio was transcoded to AAC (M4A) instead.")

            # remove audio-less video
            os.remove(targetNoAudio)
    else:
        os.remove(targetNoAudio)

    # remove temp directory
    shutil.rmtree("temp")

def run(args):
    assert (not args.video is None or not args.img is None)
    if args.skip:
        print("skip flag is abandoned, please refer to issue #207.")
    if args.UHD and args.scale==1.0:
        args.scale = 0.5
    assert args.scale in [0.25, 0.5, 1.0, 2.0, 4.0]
    if not args.img is None:
        args.png = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        if(args.fp16):
            torch.set_default_tensor_type(torch.cuda.HalfTensor)

    model = None
    try:
        try:
            try:
                from model.RIFE_HDv2 import Model
                model = Model()
                model.load_model(args.modelDir, -1)
                print("Loaded v2.x HD model.")
            except:
                from train_log.RIFE_HDv3 import Model
                model = Model()
                model.load_model(args.modelDir, -1)
                print("Loaded v3.x HD model.")
        except:
            from model.RIFE_HD import Model
            model = Model()
            model.load_model(args.modelDir, -1)
            print("Loaded v1.x HD model")
    except:
        from model.RIFE import Model
        model = Model()
        model.load_model(args.modelDir, -1)
        print("Loaded ArXiv-RIFE model")
    model.eval()
    model.device()

    if not args.video is None:
        videoCapture = cv2.VideoCapture(args.video)
        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        tot_frame = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
        videoCapture.release()
        if args.fps is None:
            fpsNotAssigned = True
            args.fps = fps * (2 ** args.exp)
        else:
            fpsNotAssigned = False
        videogen = skvideo.io.vreader(args.video)
        lastframe = next(videogen)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video_path_wo_ext, ext = os.path.splitext(args.video)
        print('{}.{}, {} frames in total, {}FPS to {}FPS'.format(video_path_wo_ext, args.ext, tot_frame, fps, args.fps))
        if args.png == False and fpsNotAssigned == True:
            print("The audio will be merged after interpolation process")
        else:
            print("Will not merge audio because using png or fps flag!")
    else:
        videogen = []
        for f in os.listdir(args.img):
            if 'png' in f:
                videogen.append(f)
        tot_frame = len(videogen)
        videogen.sort(key= lambda x:int(x[:-4]))
        lastframe = cv2.imread(os.path.join(args.img, videogen[0]), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
        videogen = videogen[1:]
    h, w, _ = lastframe.shape
    vid_out_name = None
    vid_out = None
    image_output_path = None
    if args.png:
        image_output_path = args.output
        if not os.path.exists(image_output_path):
            os.mkdir(image_output_path)
    else:
        if args.output is not None:
            vid_out_name = args.output
        else:
            vid_out_name = '{}_{}X_{}fps.{}'.format(video_path_wo_ext, (2 ** args.exp), int(np.round(args.fps)), args.ext)
        vid_out = cv2.VideoWriter(vid_out_name, fourcc, args.fps, (w, h))

    def clear_write_buffer(user_args, write_buffer):
        cnt = 0
        while True:
            item = write_buffer.get()
            if item is None:
                break
            if user_args.png:
                cv2.imwrite((image_output_path + '/{:0>7d}.png').format(cnt), item[:, :, ::-1])
                if args.callback is not None:
                    args.callback(cnt/(tot_frame * (2 ** args.exp)))
                cnt += 1
            else:
                vid_out.write(item[:, :, ::-1])

    def build_read_buffer(user_args, read_buffer, videogen):
        try:
            for frame in videogen:
                if not user_args.img is None:
                    frame = cv2.imread(os.path.join(user_args.img, frame), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
                if user_args.montage:
                    frame = frame[:, left: left + w]
                read_buffer.put(frame)
        except:
            pass
        read_buffer.put(None)

    def make_inference(I0, I1, n):
        # global model
        middle = model.inference(I0, I1, args.scale)
        if n == 1:
            return [middle]
        first_half = make_inference(I0, middle, n=n//2)
        second_half = make_inference(middle, I1, n=n//2)
        if n%2:
            return [*first_half, middle, *second_half]
        else:
            return [*first_half, *second_half]

    def pad_image(img):
        if(args.fp16):
            return F.pad(img, padding).half()
        else:
            return F.pad(img, padding)

    if args.montage:
        left = w // 4
        w = w // 2
    tmp = max(32, int(32 / args.scale))
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    padding = (0, pw - w, 0, ph - h)
    pbar = tqdm(total=tot_frame)
    if args.montage:
        lastframe = lastframe[:, left: left + w]
    write_buffer = Queue(maxsize=500)
    read_buffer = Queue(maxsize=500)
    _thread.start_new_thread(build_read_buffer, (args, read_buffer, videogen))
    _thread.start_new_thread(clear_write_buffer, (args, write_buffer))

    I1 = torch.from_numpy(np.transpose(lastframe, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
    I1 = pad_image(I1)
    temp = None # save lastframe when processing static frame

    while True:
        if temp is not None:
            frame = temp
            temp = None
        else:
            frame = read_buffer.get()
        if frame is None:
            break
        I0 = I1
        I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
        I1 = pad_image(I1)
        I0_small = F.interpolate(I0, (32, 32), mode='bilinear', align_corners=False)
        I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
        ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])

        break_flag = False
        if ssim > 0.996:        
            frame = read_buffer.get() # read a new frame
            if frame is None:
                break_flag = True
                frame = lastframe
            else:
                temp = frame
            I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
            I1 = pad_image(I1)
            I1 = model.inference(I0, I1, args.scale)
            I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
            ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
            frame = (I1[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
        
        if ssim < 0.2:
            output = []
            for i in range((2 ** args.exp) - 1):
                output.append(I0)
            '''
            output = []
            step = 1 / (2 ** args.exp)
            alpha = 0
            for i in range((2 ** args.exp) - 1):
                alpha += step
                beta = 1-alpha
                output.append(torch.from_numpy(np.transpose((cv2.addWeighted(frame[:, :, ::-1], alpha, lastframe[:, :, ::-1], beta, 0)[:, :, ::-1].copy()), (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.)
            '''
        else:
            output = make_inference(I0, I1, 2**args.exp-1) if args.exp else []

        if args.montage:
            write_buffer.put(np.concatenate((lastframe, lastframe), 1))
            for mid in output:
                mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
                write_buffer.put(np.concatenate((lastframe, mid[:h, :w]), 1))
        else:
            write_buffer.put(lastframe)
            for mid in output:
                mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
                write_buffer.put(mid[:h, :w])
        pbar.update(1)
        lastframe = frame
        if break_flag:
            break

    if args.montage:
        write_buffer.put(np.concatenate((lastframe, lastframe), 1))
    else:
        write_buffer.put(lastframe)
    import time
    while(not write_buffer.empty()):
        time.sleep(0.1)
    pbar.close()
    if not vid_out is None:
        vid_out.release()

    # move audio to new video file if appropriate
    if args.png == False and fpsNotAssigned == True and not args.video is None:
        try:
            transferAudio(args.video, vid_out_name)
        except:
            print("Audio transfer failed. Interpolated video will have no audio")
            targetNoAudio = os.path.splitext(vid_out_name)[0] + "_noaudio" + os.path.splitext(vid_out_name)[1]
            os.rename(targetNoAudio, vid_out_name)


class Args:
      def __init__(self, exp=None, img=None, output=None, png=None, callback=None):
            self.exp = exp
            self.img = img
            self.output = output
            self.png = png
            self.video = None
            self.skip = False
            self.ext = 'mp4'
            self.fps = None
            self.montage = False
            self.UHD = False
            self.scale = 1.0
            self.fp16 = False
            self.modelDir = 'ECCV2022-RIFE/train_log' 



            self.callback = callback

def run_interpolate(output_folder_path=None,
                    input_folder_path=None,
                    callback=None,):
      
      # make the output folder if it doesn't exist
      os.makedirs(output_folder_path, exist_ok=True)

      args = Args(exp=2,
                  img=input_folder_path, 
                  output=output_folder_path, 
                  png=True, 
                  callback=callback)
      run(args)

if __name__ == '__main__':
      run_interpolate(output_folder_path='outputs/interpolated/00000000',
                      input_folder_path='outputs/simple_video_sample/svd/00000000')