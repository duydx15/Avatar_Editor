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
from pydub import AudioSegment
import librosa
import soundfile as sf


# import cv2.cuda
# import cupy as cp

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def ffmpeg_encoder(outfile, fps, width, height,bitrate):
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
        # hwaccel_output_format="cuda",
        thread_queue_size=1,
        )
        encoder_ = subprocess.Popen(
        ffmpeg.compile(
            ffmpeg.output(
                frames,
                outfile,
                pix_fmt="yuv420p",
                # vcodec="libx264",
                vcodec="h264_nvenc",
                acodec="copy",
                r=fps,
                cq=17,
                maxrate=bitrate,
                minrate= bitrate,
                bufsize= "8M",
                # rc="vbr",
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
    else:
        frames = ffmpeg.input(
        "pipe:0",
        format="rawvideo",
        pix_fmt="rgb24",
        vsync="1",
        s='{}x{}'.format(width, height),
        r=fps,
        thread_queue_size=1,
        )
        encoder_ = subprocess.Popen(
        ffmpeg.compile(
            ffmpeg.output(
                frames,
                outfile,
                pix_fmt="yuv420p",
                # vcodec="libx264",
                vcodec="libx264",
                acodec="copy",
                r=fps,
                crf=17,
                maxrate=bitrate,
                minrate= bitrate,
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
    stop_last = list_stop[-1]
    list_stop = list_stop[:-1]
    list_stop.insert(0,0)
    for i in range(len(list_start)):
        if list_start[i] > list_stop[i]:
            list_stop[i] = list_start[i]
    # list_start[0] = stop_1
    list_stop.append(stop_last)
    list_stop.remove(0)
    return list_stop

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
    return cv2.add(img2_no_mask, img1_mask_only)

def add_image_by_mask_gpu(img1, img2, mask_):
    
    mask_not = cv2.cuda.bitwise_not(mask_)
    img2_no_mask = cv2.cuda.bitwise_not(img2,  mask=mask_not)
    cv2.cuda.bitwise_not(img2_no_mask, img2_no_mask, mask=mask_not)
    
    img1_mask_only = cv2.cuda.bitwise_not(img1,  mask=mask_)
    cv2.cuda.bitwise_not(img1_mask_only, img1_mask_only, mask=mask_)
    return cv2.cuda.add(img2_no_mask, img1_mask_only)#.download()

def write_frame(images,encoder_video):
    global coorCropBox
    image_draw = cv2.cvtColor(images,cv2.COLOR_RGB2BGR)
    imageout = Image.fromarray(np.uint8(image_draw))
    imageout = imageout.crop(coorCropBox)
    encoder_video.stdin.write(imageout.tobytes())


def convert_coordinates(string, single_list = False):
    # Split the string into two parts by the comma separator
    parts = ast.literal_eval(string)
    result = []
    if single_list:
        print("single input")
        parts = np.array([parts])
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
    save_path = None
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--main_video', default="")
    parser.add_argument('--godot_videos', default="")
    parser.add_argument('--timestamp', default="")
    parser.add_argument('--color_background', default="")
    parser.add_argument('--output_path', default="")
    parser.add_argument('--coordinate', default="")
    parser.add_argument('--list_audio', default="")
    parser.add_argument('--show_Avatar', default="True")
    parser.add_argument('--main_volume', default=1.0)
    parser.add_argument('--avatar_volume', default="")
    parser.add_argument('--resolution', default='720')
    parser.add_argument('--ratio', default="16:9")
    parser.add_argument('--crop_to_fit', default="True")
    
    args_tmp = parser.parse_args()
    
    show_Avatar = args_tmp.show_Avatar
    video_main_path = args_tmp.main_video
    ratio_final = args_tmp.ratio
    res_final = args_tmp.resolution
    crop_to_fit = eval(args_tmp.crop_to_fit)
    
    print("video_main_path: ",video_main_path)
    list_video_path = str(args_tmp.godot_videos).split(",")
    print("list_video_path: ",list_video_path)
    list_audio_path = str(args_tmp.list_audio).split(",")
    print("list_audio_path: ",list_audio_path)
    list_timestamp = [float(x) for x in args_tmp.timestamp.split(",")]
    print("list_timestamp: ",list_timestamp)
    
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
    
    coordinate_list = convert_coordinates(args_tmp.coordinate, len(list_video_path) ==1)
    print("coordinate_list: ", coordinate_list)
       
    # Check missing args
    lists = {
        "list_video_path":len(list_video_path),
        "list_timestamp ":len(list_timestamp),
        "bg_color_list": len(bg_color_list),
        "list_audio_path": len(list_audio_path),
        "coordinate_list": len(coordinate_list)
    }
    if not show_Avatar:
        del lists['list_video_path']
    counter = Counter(lists.values())
    most_frequent = counter.most_common()[0]
    # print("MOST ", most_frequent)
    unique_lengths = set(lists.values())
    
    if len(unique_lengths) > 1:
        different_length = next(name for name, length in lists.items() if length != most_frequent[0])
        print(f"The list {different_length} has a different length, expected: ",most_frequent[0])
        # sys.exit()
    
    main_volume = args_tmp.main_volume
    avatar_volume = args_tmp.avatar_volume
    main_volume = float(main_volume)
    
    if avatar_volume == "":
        avatar_volume = np.ones(len(list_video_path),dtype=float)
    
    elif len(list_audio_path) != len(avatar_volume.split(",")):     #Prevent number of input volume not equal to num of audio
        avatar_volume = np.ones(len(list_audio_path),dtype=float)*float(avatar_volume.split(",")[0])
    else:
        avatar_volume = [float(x) for x in avatar_volume.split(",")]
    print("Avatar_volume: ", avatar_volume, main_volume)
    
    return list_video_path,list_audio_path,video_main_path,\
            list_timestamp,bg_color_list,save_path,coordinate_list,\
            main_volume,avatar_volume,ratio_final, res_final,crop_to_fit
    
def load_godot_video():
    global cap_merge,count_godot_video,list_video_path, merge_status
    merge_status = True
    count_godot_video +=1
    # print("Video godot ",count_godot_video +1, " th")
    
def load_bg_color():
    global lower_bound, upper_bound,BG_color_list, bg_color_array
    for idx in range(len(BG_color_list)):
        bg_color_file = BG_color_list[idx]
        if not os.path.exists(bg_color_file):
            print(bg_color_file," doesn't exist!")
            bg_color =(0, 177, 64)
        else:
            with open(bg_color_file,"r") as f:
                lines = f.readlines()
                bg_color =ast.literal_eval(lines[0])
        # Convert the color to HSV format
        hsv_background_color = cv2.cvtColor(np.uint8([[bg_color]]), cv2.COLOR_RGB2HSV)
        background_color_hsv = hsv_background_color[0][0]
        # Define the range of the background color
        bg_color_array.append( bg_color)
        lower_bound.append(np.array([background_color_hsv[0] - 10, 100, 100],dtype=np.uint8))
        upper_bound.append(np.array([background_color_hsv[0] + 10, 255, 255],dtype=np.uint8))

def load_Matrix_coor():
    global width_base_video,height_base_video, video_captures, List_points, M_coor_total,ratio_final
    default_bg = [1920.0,1080.0]
    size_tmp = [1920.0,1080.0]
    #check ratio and get width, height tmp
    # if ratio_final == "16:9":
    #     size_tmp = [1920.0,1080.0]
    # elif ratio_final == "9:16":
    #     size_tmp = [608.0,1080.0]
    # elif ratio_final == "1:1":
    #     size_tmp = [1080.0,1080.0]
    
    for idx in range(len(video_captures)):
        if not video_captures[idx] is None:
            # Define old corner
            w_merge, h_merge = int(video_captures[idx].get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_captures[idx].get(cv2.CAP_PROP_FRAME_HEIGHT))
            old_corner = np.array([(0,0),(w_merge,0),(w_merge,h_merge),(0,h_merge)], np.int32)
            old_corner = old_corner.reshape(-1,1,2)
            #                   --------- default coors -------------------     ------------------------ leftovers coors -----------------------------------
            new_corners_tmp = List_points[idx]*[default_bg[0],default_bg[1]]
            # print("ori coors ",new_corners_tmp) 
            new_corners_tmp = new_corners_tmp - [float(default_bg[0]-size_tmp[0])/2,float(default_bg[1]-size_tmp[1])/2]
            # print("temp coors ",new_corners_tmp) 
            new_corners = new_corners_tmp*[width_base_video/size_tmp[0],height_base_video/size_tmp[1]] # normalized coors 
            # print("final coors ",new_corners) 
            M_coor, _ = cv2.findHomography(old_corner, new_corners)
        else:
            M_coor = None
        M_coor_total.append(M_coor)


def merge_frame_gpu(idx):
    global width_base_video, height_base_video, mask_merge, frame_merge_final, video_captures, bg_color_array
    
    if video_captures[idx] == None:
        return
    ret_merge, frame_merge = video_captures[idx].read()
    if not ret_merge:
        frame_merge = previous_frame[idx]
    frame_merge_gpu = cv2.cuda_GpuMat(frame_merge)
    # print("type: ", lower_bound[idx].tolist(),type(lower_bound[idx].tolist()), type([ 61,100, 100]))
    # mask_tmp = cv2.cuda.warpPerspective(mask_fr,M_coor_total[idx],(width_base_video, height_base_video))
    # cv2.cuda.add(mask_merge,mask_tmp ,mask_merge)
    frame_merge_tmp = cv2.cuda.warpPerspective(frame_merge_gpu,M_coor_total[idx],(width_base_video, height_base_video),borderValue=bg_color_array[idx][::-1])
    mask_fr = cv2.cuda.inRange(cv2.cuda_GpuMat(cv2.cvtColor(frame_merge_tmp.download().astype(np.uint8), cv2.COLOR_BGR2HSV)), lower_bound[idx].tolist(),upper_bound[idx].tolist())#lower_bound[idx], upper_bound[idx]) #.astype(np.uint8), cv2.COLOR_BGR2HSV)
    mask_tmp = cv2.cuda.bitwise_not(mask_fr)
    # cv2.imwrite("./Test_warp.png",frame_merge_tmp.download())
    # frame_merge_final =cv2.cuda.add(frame_merge_final,frame_merge_tmp)
    frame_merge_final = add_image_by_mask_gpu(frame_merge_tmp,frame_merge_final,mask_tmp)
    previous_frame[idx] = frame_merge

def merge_frame(idx):
    global width_base_video, height_base_video, mask_merge, frame_merge_final, video_captures
    if video_captures[idx] == None:
        return
    ret_merge, frame_merge = video_captures[idx].read()
    if not ret_merge:
        frame_merge = previous_frame[idx]
    mask_fr = cv2.inRange(cv2.cvtColor(frame_merge.astype(np.uint8), cv2.COLOR_BGR2HSV), lower_bound[idx], upper_bound[idx]) 
    mask =  cv2.bitwise_not(mask_fr)
    mask_tmp = cv2.warpPerspective(mask,M_coor_total[idx],(width_base_video, height_base_video), mask_merge,borderMode=cv2.BORDER_TRANSPARENT)
    frame_merge = cv2.bitwise_and(frame_merge,frame_merge,mask=mask)
    frame_merge_final = cv2.warpPerspective(frame_merge,M_coor_total[idx],(width_base_video, height_base_video), frame_merge_final, borderMode=cv2.BORDER_TRANSPARENT)
    previous_frame[idx] = frame_merge

def find_indices(list_to_check, item_to_find):
    return [idx for idx, value in enumerate(list_to_check) if value == item_to_find]

def update_status(status):
    global count_frame, merge_status, Timestamp_start, Timestamp_stop, video_captures
    if status:
        list_tmp = find_indices(Timestamp_start,count_frame)
        # print("Time start",count_frame, type(list_tmp), type(Timestamp_start.index(count_frame)))
        merge_status = [True if i in list_tmp else x for i, x in enumerate(merge_status)]
        print("change Start", list_tmp, merge_status)
    else:
        list_tmp = find_indices(Timestamp_stop,count_frame)
        # print("Time start",count_frame, type(list_tmp), type(Timestamp_start.index(count_frame)))
        merge_status = [False if i in list_tmp else x for i, x in enumerate(merge_status)]
        print("change stop", list_tmp, merge_status)
    
def check_res_final(ratio_, res_):
    scaled_res = [0,0] #(witdh, height)
    res_ = int(res_)
    # ratio_ = "16:9"
    if ratio_ =="" or res_ == "":
        print(f"""Empty {"ratio" if ratio_ == "" else "resolution" } input""")
        return ['',''] 
    
    if ratio_ != "16:9":
        # scaled_res[0] = int(res_)
        ratio_ = ratio_.split(':')
        ratio_ = list(map(int, ratio_))
        scaled_tmp = int(res_*ratio_[1]/ratio_[0]) 
        scaled_res[1] = scaled_tmp if ((scaled_tmp%2) ==0) else (scaled_tmp+1)
        scaled_tmp = int(scaled_tmp*16/9) 
        scaled_res[0] = scaled_tmp if ((scaled_tmp%2) ==0) else (scaled_tmp+1)
    else:
        ratio_ = ratio_.split(':')
        ratio_ = list(map(int, ratio_))
        scaled_res[1] = res_
        scaled_tmp = int(res_*ratio_[0]/ratio_[1])
        scaled_res[0] = scaled_tmp if ((scaled_tmp%2) ==0) else (scaled_tmp+1)
    return scaled_res

def getRealResFinal(ratio,res_,scaletmp):
    
    scaled_res = [0,0] #(witdh, height)
    res_ = int(res_)
    if ratio == "9:16":
        scaled_res[0] = int(res_)
        scaled_tmp = int(res_*16/9) 
        scaled_res[1] = scaled_tmp if ((scaled_tmp%2) ==0) else (scaled_tmp+1)
        # cv2.rectangle(maskTmp,((width-height*9/16)/2,0),((width+height*9/16)/2,height),(255,255,255),thickness=-1)
    else:
        ratio = ratio.split(':')
        ratio = list(map(int, ratio))
        scaled_res[1] = res_
        scaled_tmp = int(res_*ratio[0]/ratio[1])
        scaled_res[0] = scaled_tmp if ((scaled_tmp%2) ==0) else (scaled_tmp+1)
    _cropFinal = (int((scaletmp[0]-scaled_res[0])/2),0,int((scaletmp[0]-scaled_res[0])/2)+scaled_res[0],scaled_res[1])
    return scaled_res,_cropFinal

def change_audio_volume(list_audios,volumes):
    _list_audios = list_audios
    for idx in range(len(list_audios)):
        if volumes[idx] != 1.0 or list_audios[idx].endswith("mp3"):
            data, sr = librosa.load(list_audios[idx],sr=None)
            split_path = list_audios[idx].split(".")
            save_path = split_path[0] + "_reduce.wav"
            if volumes[idx] != 1.0:
                data *= volumes[idx]
            if split_path[1] == "mp3":
                data = np.transpose(data)
            sf.write(save_path, data, sr)
            _list_audios[idx] = save_path
        
    return _list_audios
    
def mix_audio(list_audio,timestamp,main_audio_path,video_length):
    
    global video_main_path_tmp
    print(video_length)
    if not os.path.exists(main_audio_path):
        sound_main = AudioSegment.silent(duration=video_length*1000,frame_rate=44100)
    else:
        sound_main = AudioSegment.from_file(main_audio_path)
    
    sound_mix = sound_main
    for idx in range(len(list_audio)):
        sound_tmp = AudioSegment.from_file(list_audio[idx])
        sound_mix = sound_mix.overlay(sound_tmp,position=timestamp[idx]*1000)
    final_audio_path = os.path.join(os.path.dirname( main_audio_path) , "final_audio.wav")
    sound_mix.export(final_audio_path, format="wav", bitrate = "192k")
    return final_audio_path

if __name__=='__main__':
   
    #Define global var
    bg_color_array = []
    lower_bound = []
    upper_bound = []
    Timestamp_stop = []
    video_captures = []
    M_coor_total = []
    mask_merge = []
    frame_merge_final = []
    bitrate_Output = None
    
    list_video_path,list_audio_path,video_main_path,list_timestamp,\
    BG_color_list,save_path,List_points,main_volume,avatar_volume,\
    ratio_final, res_final,crop_to_fit = load_cmd_input()
    # list_timestamp_tmp = [x for x, y in zip(list_timestamp, list_video_path) if y != 'null']

     # Append multiple godot videos
    for video_path in list_video_path:
        if video_path != "null":
            video_captures.append(cv2.VideoCapture(video_path))
            Timestamp_stop.append(int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT)))
        else:
            video_captures.append(None)
            Timestamp_stop.append(0)
    
    scale_output = check_res_final(ratio_final,res_final)
    scaleFinal,coorCropBox = getRealResFinal(ratio_final,res_final,scale_output)
    
    #define birate for final Video accord
    if res_final == "720":
        bitrate_Output = "5M"
    elif res_final == "1080":
        bitrate_Output = "8M"
    elif res_final == "2160":
        bitrate_Output = "40M"
    # pre-processed video according ratio
    if ratio_final == "16:9":
        preprocessed_size = [1920,1080]
    elif ratio_final == "9:16":
        preprocessed_size = [608,1080]
    elif ratio_final == "1:1":
        preprocessed_size = [1080,1080]
         
    #preprocessing main video
    video_preprocessed = video_main_path.split(".")[0] + "_preprocess.mp4"
    video_main_path_tmp = video_main_path.split(".")[0] + "_tmp.mp4"
    
    # prepare -vf sequence for scaling video
    if crop_to_fit:
        vf_param = f"""scale={scale_output[0]}:{scale_output[1]}:force_original_aspect_ratio=1, pad={scale_output[0]}:{scale_output[1]}:-1:-1, setsar=1"""
        vf_preprocessed_param = f"""scale={preprocessed_size[0]}:{preprocessed_size[1]}:force_original_aspect_ratio=increase, crop={preprocessed_size[0]}:{preprocessed_size[1]}"""
        video_preprocessed = video_main_path
    else:
        vf_param = f"""scale={scale_output[0]}:{scale_output[1]}:force_original_aspect_ratio=1, pad={scale_output[0]}:{scale_output[1]}:-1:-1, setsar=1"""
        vf_preprocessed_param = f"""scale={preprocessed_size[0]}:{preprocessed_size[1]}:force_original_aspect_ratio=1, pad={preprocessed_size[0]}:{preprocessed_size[1]}:-1:-1, setsar=1"""

    if torch.cuda.is_available():
        if scale_output[0] == '':
            ffmpeg_cmd_main_video_tmp = f"""sudo  /home/ubuntu/anaconda3/envs/gazo/bin/ffmpeg  -hwaccel_device 0 -hwaccel cuda -y -i {video_main_path}  -filter_complex fps=25 -cq:v 17 -maxrate:v {bitrate_Output} -minrate:v {bitrate_Output} -bufsize:v 8M -vcodec hevc_nvenc {video_main_path_tmp} """
        else:
            ffmpeg_cmd_main_video_tmp = f"""sudo  /home/ubuntu/anaconda3/envs/gazo/bin/ffmpeg  -hwaccel_device 0 -hwaccel cuda -y -i {video_preprocessed} -vf "{vf_param},fps=25" -cq:v 17 -maxrate:v {bitrate_Output} -minrate:v {bitrate_Output} -bufsize:v 8M -c:a copy -vcodec hevc_nvenc {video_main_path_tmp} """
            ffmpeg_preprocessed = f"""sudo  /home/ubuntu/anaconda3/envs/gazo/bin/ffmpeg  -hwaccel_device 0 -hwaccel cuda -y -i {video_main_path} -vf "{vf_preprocessed_param},fps=25" -cq:v 17 -maxrate:v {bitrate_Output} -minrate:v {bitrate_Output} -bufsize:v 8M -c:a copy -vcodec hevc_nvenc {video_preprocessed} """#
    else:
        if scale_output[0] == '':
            ffmpeg_cmd_main_video_tmp = f""" sudo  /home/ubuntu/anaconda3/envs/gazo/bin/ffmpeg  -y -i {video_main_path} -filter_complex fps=25 -crf 17 -maxrate:v {bitrate_Output} -minrate:v {bitrate_Output} -vcodec h264 {video_main_path_tmp} """
        else:
            ffmpeg_cmd_main_video_tmp = f"""  sudo  /home/ubuntu/anaconda3/envs/gazo/bin/ffmpeg  -y -i {video_preprocessed} -vf "{vf_param},fps=25" -crf 17 -maxrate:v {bitrate_Output} -minrate:v {bitrate_Output} -c:a copy -vcodec h264 {video_main_path_tmp} """
            ffmpeg_preprocessed = f""" sudo  /home/ubuntu/anaconda3/envs/gazo/bin/ ffmpeg  -y -i {video_main_path} -vf "{vf_preprocessed_param},fps=25" -crf 17 -maxrate:v {bitrate_Output} -minrate:v {bitrate_Output} -c:a copy -vcodec h264 {video_preprocessed} """
    
    if scale_output[0] != '' and not crop_to_fit:
        print("FFMPEG COMMAND: ",ffmpeg_preprocessed)
        os.system(ffmpeg_preprocessed)
        time.sleep(2)        
    
    print("FFMPEG COMMAND: ",ffmpeg_cmd_main_video_tmp)
    os.system(ffmpeg_cmd_main_video_tmp)
    time.sleep(1)
    
    cap = cv2.VideoCapture(video_main_path_tmp)
    # cap = cv2.VideoCapture(video_main_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width_base_video = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_base_video = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # print("#################", width_base_video, height_base_video)
    path_folder = os.path.dirname(os.path.abspath(save_path))
    output_nonsound = os.path.join(path_folder,"output_nonsound.mp4")
    # Check exist output_nonsound.mp4
    if os.path.exists(output_nonsound):
        os.remove(output_nonsound)
    encoder_video = ffmpeg_encoder(output_nonsound, fps,scaleFinal[0], scaleFinal[1],bitrate_Output)
    
    # #Process timestamp
    Timestamp_start = [round(fps*x) for x in list_timestamp]
    cap_merge = None    
    # Load data merge
    load_bg_color()
    load_Matrix_coor()
    # Define the background color to be removed
    Timestamp_stop = (np.array(Timestamp_stop) + np.array(Timestamp_start)).tolist()
    print("Timestamp_start: ", Timestamp_start)
    print("Timestamp_stop: ", Timestamp_stop)
    Timestamp_stop = fine_tune_timestamp(Timestamp_start,Timestamp_stop)
    print("Timestamp_stop_finetuned: ", Timestamp_stop)
    
    # Define count vars
    count_frame = 0
    count_godot_video = 0
    merge_status = [False] * len(list_video_path)
    previous_frame = [None] * len(list_video_path)
    tqdm = tqdm(total=total_frames)

    while cap.isOpened():
        
        ret, frame = cap.read()
        if not  ret:
            break   
        # Update status merge for each video
        if count_frame in Timestamp_start:
            update_status(True)
        if count_frame in Timestamp_stop:
            update_status(False)
        # print(any(merge_status)
        if any(merge_status) is True:
            
            if torch.cuda.is_available(): 
                frame_merge_final = cv2.cuda_GpuMat(frame)#np.zeros((height_base_video,width_base_video,3),dtype=np.uint8))
                # get list index of merge video are True
                list_idx = np.where(merge_status)[0]
                for idx in list_idx:
                    merge_frame_gpu(idx)
                output_main = frame_merge_final.download()
            else:
                frame_merge_final = frame.copy()
                mask_merge = np.zeros((height_base_video,width_base_video),dtype=np.uint8)
                # get list index of merge video are True
                list_idx = np.where(merge_status)[0]
                for idx in list_idx:
                    merge_frame(idx)
                output_main = add_image_by_mask(frame_merge_final,frame,mask_merge)
            write_frame(output_main,encoder_video)
            tqdm.update(1)
            count_frame +=1
            
        else:
            #write frame base video without any merge
            write_frame(frame,encoder_video)
            tqdm.update(1)
            count_frame +=1
            
    cap.release()
    encoder_video.stdin.flush()
    encoder_video.stdin.close()
    for cap in video_captures:
        if cap != None:
            cap.release()
    time.sleep(1)
    # Extract main audio
    main_audio = os.path.join(path_folder,"main_audio.wav")
    # Check exist main_audio.mp4
    if os.path.exists(main_audio):
        os.remove(main_audio)
    os.system(f""" ffmpeg  -y -i {video_main_path} -q:a 0 -map a {main_audio}""")
    # Merge audio
    time.sleep(1)

    list_audio_path_mix = change_audio_volume(list_audio_path,avatar_volume) 
    path_final_audio = mix_audio(list_audio_path_mix,list_timestamp,main_audio,float(total_frames/fps))
    print("##################### List", list_audio_path_mix)
    try:
        ffmpeg_cmd_merge_audio = f"""sudo  /home/ubuntu/anaconda3/envs/gazo/bin/ffmpeg -y -i {output_nonsound} -i {path_final_audio} -c:v copy -c:a libmp3lame -b 192k {save_path}"""
        print("FFMPEG COMMAND: ",ffmpeg_cmd_merge_audio)
        os.system(ffmpeg_cmd_merge_audio)
        print("######### COMPLETED  #########")
        time.sleep(1)
        # os.remove(output_nonsound)        
    except Exception:
        print("Can not merge audio to output_video")
        os.rename(output_nonsound,save_path)
        # /home/ubuntu/anaconda3/envs/gazo/bin/