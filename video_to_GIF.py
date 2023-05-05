import cv2
import numpy as np
import ffmpeg
import subprocess
from tqdm import tqdm
import ast
import sys
from PIL import Image
import os
import argparse
import time 
from collections import Counter
import torch


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def ffmpeg_encoder(outfile, fps, width, height):
    LOGURU_FFMPEG_LOGLEVELS = {
        "trace": "trace",
        "debug": "debug",
        "info": "info",
        "success": "info",
        "warning": "warning",
        "error": "error",
        "critical": "fatal",
        }


    if torch.cuda.is_available():
        frames = ffmpeg.input(
        "pipe:0",
        format="rawvideo",
        pix_fmt="rgb24",
        vsync="1",
        s='{}x{}'.format(width, height),
        r=fps,
        hwaccel="cuda",
        hwaccel_device="0",
        hwaccel_output_format="cuda",
        # thread_queue_size=1,
    )
        codec = "h264_nvenc"
    else:
        frames = ffmpeg.input(
        "pipe:0",
        format="rawvideo",
        pix_fmt="rgb24",
        vsync="1",
        s='{}x{}'.format(width, height),
        r=fps,
        # hwaccel="cuda",
        # hwaccel_device="0",
        # hwaccel_output_format="cuda",
        # thread_queue_size=1,
    )
        codec = "libx264"
    # print("###########33", codec)
    encoder_ = subprocess.Popen(
        ffmpeg.compile(
            ffmpeg.output(
                frames,
                outfile,
                pix_fmt="yuv420p",
                # vcodec="libx264",
                vcodec=codec,
                acodec="copy",
                r=fps,
                crf=17,
                vsync="1",
                # async=4,
            )
            .global_args("-hide_banner")
            .global_args("-nostats")
            .global_args(
                "-loglevel",
                LOGURU_FFMPEG_LOGLEVELS.get(
                    os.environ.get("LOGURU_LEVEL", "INFO").lower()
                ),
            ),
            overwrite_output=True,
        ),
        stdin=subprocess.PIPE,
        # stdout=subprocess.DEVNULL,
        # stderr=subprocess.DEVNULL,
    )
    return encoder_

def fine_tune_timestamp(list_start,list_stop):
    list_stop = list_stop[:-1]
    list_stop.insert(0,0)
    for i in range(len(list_start)):
        if list_start[i] <= list_stop[i]:
            list_start[i] = list_stop[i]+2
    return list_start

def mix_pixel(pix_1, pix_2, perc):
    return (perc/255 * pix_1) + ((255 - perc)/255 * pix_2)

# function for blending images depending on values given in mask
def blend_images_using_mask(img_orig, img_for_overlay, img_mask):
    if len(img_mask.shape) != 3:
        img_mask = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)
    img_res = mix_pixel(img_orig, img_for_overlay, img_mask)
    return img_res.astype(np.uint8)

def add_image_by_mask(img1, img2, mask_):
    mask_not = cv2.bitwise_not(mask_)
    img2_no_mask = cv2.bitwise_and(img2, img2, mask=mask_not)
    img1_mask_only = cv2.bitwise_and(img1, img1, mask=mask_)
    return cv2.add(img2_no_mask, img1_mask_only,dtype=cv2.CV_64F)

def write_frame(images,encoder_video):
    image_draw = cv2.cvtColor(images,cv2.COLOR_RGB2BGR)
    imageout = Image.fromarray(np.uint8(image_draw))
    encoder_video.stdin.write(imageout.tobytes())


def convert_coordinates(string, single_list = False):
    # Split the string into two parts by the comma separator
    parts = ast.literal_eval(string)
    result = []
    if single_list:
        print("single input")
        parts = np.array([parts])
    print("Length: ", parts)
    # Loop through the two parts
    for part in parts:
        # Split each part into its individual coordinates
        coords = part#.replace("(","").replace(")","").replace(" ","").split(",")
        # Convert the coordinates into tuples
        tuples = [(float(coords[i]), float(coords[i+1])) for i in range(0, len(coords), 2)]
        # Add the tuples to the result list
        result.append(tuples)
    
    return np.array(result)

def load_cmd_input():
    list_video_path = None
    video_main_path = None
    list_timestamp = None
    list_rotation = None
    save_path = None
    list_scale = None
    list_position = None
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--main_video', default="")
    parser.add_argument('--color_background', default="")
    parser.add_argument('--output_path', default="")
    args_tmp = parser.parse_args()
    
    video_main_path = args_tmp.main_video
        
    print("video_main_path: ",video_main_path)
    
    # Change perrmission output folder
    save_path = args_tmp.output_path
    print("save_path: ", save_path)
    
    path_folder = os.path.dirname(os.path.abspath(save_path))
    if not os.path.exists(path_folder):
        os.umask(0)
        os.makedirs(path_folder,mode=0o777)
    # else:
    #     os.chmod(path_folder,mode=0o444)
    bg_color_list = str(args_tmp.color_background).split(",")
    print("bg_color_list: ", bg_color_list)
    
    
    return video_main_path,bg_color_list,save_path
    
def find_other_corners(top_left, width, height, X):
    # angle = X * np.pi / 180
    a = top_left
    angle = np.deg2rad(-X) # Convert angle to radians
    top_right = (top_left[0] + width * np.cos(angle), top_left[1] + width * np.sin(angle))
    bottom_right = (top_right[0] - height * np.sin(angle), top_right[1] + height * np.cos(angle))
    bottom_left = (top_left[0] - height * np.sin(angle), top_left[1] + height * np.cos(angle))
    
    return [a,top_right, bottom_right, bottom_left]

def load_godot_video():
    global cap_merge,count_godot_video,list_video_path, merge_status
    merge_status = True
    count_godot_video +=1
    print("Video godot ",count_godot_video +1, " th")
    

if __name__=='__main__':
   
    video_main_path,BG_color_list,save_path = load_cmd_input()
    
     #preprocessing main video
    video_main_path_tmp = video_main_path.split(".")[0] + "_tmp.mp4"
    
    cap = cv2.VideoCapture(video_main_path)
    # cap = cv2.VideoCapture(video_main_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    path_folder = os.path.dirname(os.path.abspath(save_path))
    output_nonsound = os.path.join(path_folder,"output_nonsound.mp4")
    # Check exist output_nonsound.mp4
    if os.path.exists(output_nonsound):
        os.remove(output_nonsound)
    encoder_video = ffmpeg_encoder(output_nonsound, fps,width, height)
    
    # Define count vars
    count_frame = 0
    count_godot_video = 0
    merge_status = False
    tqdm = tqdm(total=total_frames)
    
    bg_color_file = BG_color_list[count_godot_video]
    # print("bg_color_list[count_godot_video] ",BG_color_list[count_godot_video])
    if not os.path.exists(bg_color_file):
        print(bg_color_file," doesn't exist!")
        bg_color =(0, 177, 64)
        # sys.exit()
    else:
        with open(bg_color_file,"r") as f:
            lines = f.readlines()
            bg_color =ast.literal_eval(lines[0])
    # Convert the color to HSV format
    while cap.isOpened():
        ret, frame = cap.read()
        if not  ret:
            break
        hsv_background_color = cv2.cvtColor(np.uint8([[bg_color]]), cv2.COLOR_RGB2HSV)
        background_color_hsv = hsv_background_color[0][0]
        lower_bound = np.array([background_color_hsv[0] - 10, 100, 100])
        upper_bound = np.array([background_color_hsv[0] + 10, 255, 255])
        mask = cv2.inRange(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2HSV), lower_bound, upper_bound)
        mask =  cv2.bitwise_not(mask)
        mask = cv2.GaussianBlur(mask,(1,1),0)
        write_frame(mask,encoder_video)
        tqdm.update(1)
    cap.release()
    encoder_video.stdin.flush()
    encoder_video.stdin.close()
    
    #Convert to gif
    time.sleep(1)
    os.system(f""" ffmpeg -y -i {video_main_path} -i {output_nonsound} -filter_complex "[1][0]scale2ref[mask][main];[main][mask]alphamerge,fps={fps},split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" {save_path}""")
            
    print("######### COMPLETED  #########")
    # time.sleep(1)
    os.remove(output_nonsound)        