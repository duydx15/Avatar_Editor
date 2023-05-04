from PIL import Image
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', default="")
    parser.add_argument('--save_color_file', default="")
    args_tmp = parser.parse_args()
    
    input_image = args_tmp.input_image
    output_file = args_tmp.save_color_file
    
    if not os.path.exists(input_image):
        print("Input image doesn't exist!")
        exit()
    path_folder = os.path.dirname(os.path.abspath(output_file))
    if not os.path.exists(path_folder):
        os.umask(0) 
        os.makedirs(path_folder,mode=0o777)
    # output_file = os.path.join(output_folder,"unique_color.txt")
    # Load the image
    img = Image.open(str(input_image)).convert('RGB')

    # Get the image size
    width, height = img.size
    # Loop through the pixels and find a unique color
    background_color = (0, 0, 0)
    for i in range(width):
        for j in range(height):
            pixel = img.getpixel((i, j))
            if pixel != background_color:
                background_color = (0,pixel[1],0)
                break
        if background_color != (0, 0, 0):
            break

    print("Unique color", background_color)
    if background_color == (0,0,0):
        print("Could not find unique color! Auto set unique color = (0,255,0)")
        background_color = (0,255,0)
    f = open(output_file, 'w')
    f.write(str(background_color))
    f.close()
    
if __name__=='__main__':
    main()