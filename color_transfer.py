import argparse
import cv2
import numpy as np
import os


def mean_std_transfer(img_arr_in, mask_in, img_arr_ref, mask_ref):
    """
    https://github.com/pengbo-learn/python-color-transfer/blob/master/python_color_transfer/color_transfer.py#L51
    """
    masked_in = np.ma.masked_where(mask_in,img_arr_in )
    masked_ref = np.ma.masked_where(mask_ref, img_arr_ref)
    mean_in = np.mean(masked_in, axis=(0, 1), keepdims=True)
    mean_ref = np.mean(masked_ref, axis=(0, 1), keepdims=True)
    std_in = np.std(masked_in, axis=(0, 1), keepdims=True)
    std_ref = np.std(masked_ref, axis=(0, 1), keepdims=True)
    img_arr_out = (img_arr_in - mean_in) / std_in * std_ref + mean_ref
    img_arr_out[img_arr_out < 0] = 0
    img_arr_out[img_arr_out > 255] = 255
    return img_arr_out.astype("uint8")


def lab_transfer(img_arr_in, mask_in, img_arr_ref, mask_ref):
    """
    https://github.com/pengbo-learn/python-color-transfer/blob/master/python_color_transfer/color_transfer.py#L35
    """
    lab_in = cv2.cvtColor(img_arr_in, cv2.COLOR_BGR2LAB)
    lab_ref = cv2.cvtColor(img_arr_ref, cv2.COLOR_BGR2LAB)
    lab_out = mean_std_transfer(img_arr_in, mask_in, img_arr_ref, mask_ref)
    img_arr_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)
    return img_arr_out


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Lip Detection')
    parser.add_argument('dest_path', help='Absolute or relative path of destination image file', type=str)
    parser.add_argument('dest_mask_path', help='Absolute or relative path of destination mask file', type=str)
    parser.add_argument('source_path', help='Absolute or relative path of source image file', type=str)
    parser.add_argument('source_mask_path', help='Absolute or relative path of source mask file', type=str)
    parser.add_argument('output_path', help='Absolute or relative path of output file', type=str)
    args = parser.parse_args()
    dest_image = cv2.imread(args.dest_path)
    dest_mask = cv2.imread(args.dest_mask_path)
    source_image = cv2.imread(args.source_path)
    source_mask = cv2.imread(args.source_mask_path)
    transfer_image = lab_transfer(dest_image, dest_mask, source_image, source_mask)
    dest_image[np.where(dest_mask == 255)] = transfer_image[np.where(dest_mask == 255)]
    cv2.imwrite(args.output_path, dest_image)
