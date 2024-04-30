import argparse
import cv2
import mediapipe as mp
import numpy as np
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw


FACE_POINTS = {
    'lipsUpperOuter': [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
    'lipsLowerOuter': [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
    'lipsUpperInner': [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
    'lipsLowerInner': [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Lip Detection')
    parser.add_argument('input_file_path', help='Absolute or relative path of image file', type=str)
    parser.add_argument('output_file_path', help='Absolute or relative path of output file', type=str)
    args = parser.parse_args()
    if not os.path.exists(args.input_file_path):
        print('File "%s" does not exist.' % args.input_file_path)
        exit(0)
    base_options = python.BaseOptions(
        model_asset_path=os.path.join('mediapipe_tasks', 'face_landmarker_v2_with_blendshapes.task')
    )
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)
    input_image = mp.Image.create_from_file(args.input_file_path)
    detection_result = detector.detect(input_image)
    output_image = Image.new('RGB', (input_image.width, input_image.height), color='black')
    outer_lip_indices = FACE_POINTS['lipsUpperOuter'][:-1] + list(reversed(FACE_POINTS['lipsLowerOuter']))
    inner_lip_indices = FACE_POINTS['lipsUpperInner'][:-1] + list(reversed(FACE_POINTS['lipsLowerInner']))
    for face_info in detection_result.face_landmarks:
        outer_xys = []
        for outer_lip_index in outer_lip_indices:
            outer_xys.append((
                round(face_info[outer_lip_index].x * input_image.width),
                round(face_info[outer_lip_index].y * input_image.height)
            ))
        inner_xys = []
        for inner_lip_index in inner_lip_indices:
            inner_xys.append((
                round(face_info[inner_lip_index].x * input_image.width),
                round(face_info[inner_lip_index].y * input_image.height)
            ))
        draw = ImageDraw.Draw(output_image)
        draw.polygon(outer_xys, fill='white', outline='white', width=1)
        draw.polygon(inner_xys, fill='black', outline='white', width=1)
    output_image.save(args.output_file_path)
