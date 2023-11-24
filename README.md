# svd-playground

This repo is for Stable Video Diffusion (SVD) related experiments.

## Setup

We need to install all the requirements for the submodules. 

First create a virtual environment:

```
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Download the SVD checkpoints huggingface and place them in the `checkpoints` folder.

To see the options for options for the `main.py` run

```
python main.py --help
```

For a 3090 GPU you'll want to run with around 14 frames. Here is an example command:

```
python main.py --num_frames 14 --input_path=assets/init.jpg
```

This will generate frames and video in the outputs folder. 

### RIFE

By default the `main.py` uses RIFE for interpolation. To use RIFE, download the pretrained models from [here](https://drive.google.com/file/d/1APIzVeI-4ZZCEuIRE1m6WYfSCaOsi_7_/view?usp=sharing). Unzip the files and put them in the `ECCV2020-RIFE/train_log` folder.

