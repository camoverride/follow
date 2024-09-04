# Eyes


## Install

- `python -m venv .venv`
- `source .venv/bin/activate`
- `pip install -r requirements.txt`


## Data Creation

1) Use `delete_sclera.py` to remove the sclera from the eyeball video and save the frames. Default video name is `cam_face.mp4`

2) Use `visual_crop.py` to crop to the eyeball of choice.

3) Find or create a good eyeball image. Experiment with the location and cropping of this. Default image is `eyeball.png`

4) Use `animate.py` to test out the animation.


## Run in Production

TODO: add info about system d


## TODO:

- set screen size
- experient with image size, number of images, etc
- test on Pi 4B
- change model to YOLO (or other object detection algo) running on TPU
