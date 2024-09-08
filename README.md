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

- get better eye video
- get better eyeball
- install on Pi 5
- two eye version
- if slow or glitchy: pi5 can handle lots of things in memory, so remove memmap portion
- change model to YOLO (or other object detection algo) running on TPU