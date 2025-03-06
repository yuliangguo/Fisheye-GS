import os
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
from scene.colmap_loader import read_intrinsics_binary


def read_intrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
    return camera_id, model, width, height, params


def colmap_main(args):
    if args.cross_camera:
        if 'fisheye' in args.camera_path:
            args.camera_path = args.camera_path.replace('fisheye', 'undistorted')
        else:
            args.camera_path = args.camera_path.replace('undistorted', 'fisheye')
    camera_dir = Path(args.camera_path)
    input_image_dir = Path(args.src)
    out_image_dir = Path(args.dst)
    
    cam_intrinsics = read_intrinsics_binary(camera_dir)
    width = cam_intrinsics[1].width
    height = cam_intrinsics[1].height
    params = cam_intrinsics[1].params
    print(params)
    
    # adjust fx, fy, cx, cy by the actual image size
    if args.r == -1:
        ratio = 1 / float(input_image_dir.name[7])
    else:
        ratio = 1 / args.r
    
    fx = params[0] * ratio
    fy = params[1] * ratio
    cx = params[2] * ratio
    cy = params[3] * ratio
    width = int(width * ratio)
    height = int(height * ratio)
    
    distortion_params = params[4:]
    kk = distortion_params
    
    # TODO: this remapping not exact, because in undistortion, r was calculated from different domain's theta
    # Reverse warping
    reverse_mapx = np.zeros((width, height), dtype=np.float32)
    reverse_mapy = np.zeros((width, height), dtype=np.float32)
    for i in tqdm(range(0, width), desc="calculate_reverse_maps"):
        for j in range(0, height):
            x = float(i)
            y = float(j)
            x1 = (x - cx) / fx
            y1 = (y - cy) / fy
            # Inaccurcy from not easy to theta from theta_d computed from source KB space
            l = np.sqrt(x1**2 + y1**2)
            theta = np.arctan(l)
            x2 = fx * x1 * theta / l + width // 2
            y2 = fy * y1 * theta / l + height // 2
            reverse_mapx[i, j] = x2
            reverse_mapy[i, j] = y2
    
    frames = os.listdir(input_image_dir)

    for frame in tqdm(frames, desc="frame"):
        image_path = Path(input_image_dir) / frame
        undistorted_image = cv2.imread(str(image_path))
        
        reversed_image = cv2.remap(
            undistorted_image,
            reverse_mapx.T,
            reverse_mapy.T,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        reversed_image_path = Path(out_image_dir) / frame
        reversed_image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(reversed_image_path), reversed_image)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--camera-path', type=str, default="/mnt/data_ssd_4tb/Datasets/zipnerf/undistorted/berlin/sparse/0/cameras.bin")
    parser.add_argument('--src', type=str, default="/mnt/data_ssd_4tb/Datasets/zipnerf/undistorted/berlin/images_4_equidist")
    parser.add_argument('--dst', type=str, default="/mnt/data_ssd_4tb/Datasets/zipnerf/undistorted/berlin/images_4_equidist")
    parser.add_argument('-r', type=int, default=-1)
    parser.add_argument('--cross_camera', action='store_true')
    args = parser.parse_args()
    colmap_main(args)
