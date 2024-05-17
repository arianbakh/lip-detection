# Setup

- `conda create -n lip-detection python=3.11.5`
- `conda activate lip-detection` (`source activate lip-detection` on Mac)
- `pip install mediapipe==0.10.11`
- `pip install protobuf==4.25.3` (because of [this issue](https://github.com/google/mediapipe/issues/5188#issuecomment-2080437250), even though this version is technically not compatible with Mediapipe)
- `wget -O mediapipe_tasks/face_landmarker_v2_with_blendshapes.task -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`

# Example Usage

- `python3 detect.py images/1.jpeg images/1_out.jpeg`
- `python3 color_transfer.py images/1.jpeg images/1_out.jpeg images/6.jpeg images/6_out.jpeg images/1_transfer.jpeg`

# Notes

- Information about face keypoints can be found [here](https://github.com/tensorflow/tfjs-models/blob/838611c02f51159afdd77469ce67f0e26b7bbb23/face-landmarks-detection/src/mediapipe-facemesh/keypoints.ts), [here](https://storage.googleapis.com/mediapipe-assets/documentation/mediapipe_face_landmark_fullsize.png) and [here](https://raw.githubusercontent.com/google/mediapipe/a908d668c730da128dfa8d9f6bd25d519d006692/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png)
