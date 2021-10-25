import glob
import re
from PIL import Image, ImageOps
import numpy as np
from tqdm import tqdm
import cv2
import os
import argparse


def _get_sorted_images(images_path):
    images = glob.glob(images_path)
    images.sort(key=os.path.getmtime) 
    return images


def _maximum_stack(image1, image2, image1_gray=None):
    if image1_gray is None:
        image1_gray = np.expand_dims(
            np.array(ImageOps.grayscale(image1)), axis=2)
    image2_gray = np.expand_dims(
        np.array(ImageOps.grayscale(image2)), axis=2)
    image2_np = np.array(image2)
    
    res_image = np.where(image1_gray > image2_gray, image1, image2_np)
    res_gray = np.max(np.stack([image1_gray, image2_gray], axis=2), axis=2)
    return res_image, res_gray


def star_trail(images_path, save_fname=None):
    images = _get_sorted_images(images_path)
    assert len(images) > 2, "not enough images received"
    with Image.open(images[0]) as im:
        res_image = np.array(im)
        res_gray = np.expand_dims(np.array(ImageOps.grayscale(im)), axis=2)
    
    for next_image in tqdm(images[1:]):
        with Image.open(next_image) as nim:
            res_image, res_gray = _maximum_stack(res_image, nim, res_gray)
    res_image = Image.fromarray(res_image)
    if save_fname is not None:
        res_image.save(save_fname)
    return res_image


def time_lapse(images_path, save_fname=None, fps=24, out_size=None, accumulate=False):
    images = _get_sorted_images(images_path)
    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    if save_fname is None:
        save_fname = images[0][:-4] + "_time_lapse.mp4"
    try:
        frame_w = int(out_size[0])
        frame_h = int(out_size[1])
    except Exception as e:
        img = cv2.imread(images[0])
        frame_h, frame_w = img.shape[:2]
        if out_size is not None:
            print(f"Failed to understand the output video resolution {out_size}.")
        print(f"Video size: W={frame_w} H={frame_h}")
    print(f"Creating video file: {save_fname}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  #(*'avc1')
    out = cv2.VideoWriter(save_fname,
                          fourcc, 
                          fps, 
                          (frame_w,frame_h))
    if accumulate:
        im0 = Image.open(images[0])
        last_img = np.array(im0)
        last_gray = np.expand_dims(np.array(ImageOps.grayscale(im0)), axis=2)
        for fname in tqdm(images[1:], desc='writing frames'):
            last_img, last_gray = _maximum_stack(last_img, Image.open(fname), last_gray)
            frame = cv2.cvtColor(np.array(last_img), cv2.COLOR_RGB2BGR)
            img = cv2.resize(frame, (frame_w, frame_h))
            out.write(frame)
    else:
        for fname in tqdm(images, desc='writing frames'):
            img = cv2.resize(cv2.imread(fname), (frame_w, frame_h))
            out.write(img) 

    out.release()

    # Closes all the frames
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, help='Regex to the images')
    parser.add_argument('--star_trail', action='store_true', help='add this flag to produce star trail image')
    parser.add_argument('--time_lapse', action='store_true', help='add this flag to produce time lapse video')
    parser.add_argument('--trace', action='store_true', help='add this flag with --time_laspe to produce time laspe with accumulative star trails')
    parser.add_argument('--output_file', type=str, help='output file name')
    parser.add_argument('--fps', type=int, default=24, help='video fps')
    parser.add_argument('--output_resolution', nargs=2, type=int, help='output video resolution')
    args = parser.parse_args()
    try:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    except FileNotFoundError:
        pass
    if args.star_trail:
        star_trail(f"{args.images}/*.JPG", args.output_file)
    else:
        time_lapse(f"{args.images}/*.JPG", args.output_file,
         fps=args.fps, out_size=args.output_resolution, accumulate=args.trace)
