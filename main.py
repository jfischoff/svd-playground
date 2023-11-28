import math
import os
from glob import glob
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from einops import rearrange, repeat
from fire import Fire
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import ToTensor
from itertools import count
from src.video_utils import frames_to_video
from src.rife_interpolate import run_interpolate
import src.sdxl_runner as SDXL

from sgm.util import append_dims
from sgm.inference.helpers import embed_watermark
from sgm.util import default, instantiate_from_config
from sgm.inference.helpers import Img2ImgDiscretizationWrapper

def save_sampled_images(samples, output_folder):
    """
    Saves given samples as images in separate folders within the specified output folder.
    Each folder is named with a zero-padded number.

    :param samples: The samples to be saved as images.
    :param output_folder: The directory where the images will be saved.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Count the number of folders in the output folder
    base_count = sum(os.path.isdir(os.path.join(output_folder, d)) for d in os.listdir(output_folder))

    samples = embed_watermark(samples)
    images = (
        (rearrange(samples, "f c h w -> f h w c") * 255)
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    print("images.shape", images.shape)

    image_folder = os.path.join(output_folder, f"{base_count:08d}")

    for i, frame in zip(count(base_count), images):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Create a new folder for each image
        
        os.makedirs(image_folder, exist_ok=True)
        
        # Save the image in the new folder
        image_path = os.path.join(image_folder, f"{i:04d}.png")
        cv2.imwrite(image_path, frame)

    return image_folder

def sample(
    input_path = None,  # Can either be image file or folder with image files
    prompt = None,
    num_frames: Optional[int] = None,
    num_steps: Optional[int] = None,
    version: str = "svd",
    fps_id: int = 6,
    motion_bucket_id: int = 127,
    cond_aug: float = 0.02,
    seed: int = 23,
    decoding_t: int = 1,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    device: str = "cuda",
    output_folder: Optional[str] = None,
    interpolate: bool = True,
    base_model="stabilityai/stable-diffusion-xl-base-1.0",
    refiner="stabilityai/stable-diffusion-xl-refiner-1.0",
    height=576,
    width=1024,
    render_height=1152,
    render_width=2048,
    prompts="",
    negative_prompts = "(deformediris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
    dtype=torch.float32,
    num_inference_steps = 20,
    high_noise_frac = 0.8,  
    seeds=42,
    initial_video_path=None,
    img2img_strength=None,
):
    """
    Simple script to generate a single sample conditioned on an image `input_path` or multiple images, one for each
    image file in folder `input_path`. If you run out of VRAM, try decreasing `decoding_t`.
    """

    if version == "svd":
        num_frames = default(num_frames, 14)
        num_steps = default(num_steps, 25)
        output_folder = default(output_folder, "outputs/simple_video_sample/svd/")
        model_config = "configs/svd.yaml"
    elif version == "svd_xt":
        num_frames = default(num_frames, 25)
        num_steps = default(num_steps, 30)
        output_folder = default(output_folder, "outputs/simple_video_sample/svd_xt/")
        model_config = "configs/svd_xt.yaml"
    elif version == "svd_image_decoder":
        num_frames = default(num_frames, 14)
        num_steps = default(num_steps, 25)
        output_folder = default(
            output_folder, "outputs/simple_video_sample/svd_image_decoder/"
        )
        model_config = "configs/svd_image_decoder.yaml"
    elif version == "svd_xt_image_decoder":
        num_frames = default(num_frames, 25)
        num_steps = default(num_steps, 30)
        output_folder = default(
            output_folder, "outputs/simple_video_sample/svd_xt_image_decoder/"
        )
        model_config = "configs/svd_xt_image_decoder.yaml"
    else:
        raise ValueError(f"Version {version} does not exist.")
    
    # if the input_path and prompt are None raise
    if input_path is None and prompts is None:
        raise ValueError("Must provide either input_path or prompt")



    all_images = []
    if input_path is not None:
      path = Path(input_path)
      all_img_paths = []
      if path.is_file():
          if any([input_path.endswith(x) for x in ["jpg", "jpeg", "png"]]):
              all_img_paths = [input_path]
          else:
              raise ValueError("Path is not valid image file.")
      elif path.is_dir():
          all_img_paths = sorted(
              [
                  f
                  for f in path.iterdir()
                  if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
              ]
          )
          if len(all_img_paths) == 0:
              raise ValueError("Folder does not contain any images.")
      else:
          raise ValueError
      

      for input_img_path in all_img_paths:
          with Image.open(input_img_path) as image:
              
              image = image.convert("RGB")
              w, h = image.size

              if h % 64 != 0 or w % 64 != 0:
                  width, height = map(lambda x: x - x % 64, (w, h))
                  image = image.resize((width, height))
                  print(
                      f"WARNING: Your image is of size {h}x{w} which is not divisible by 64. We are resizing to {height}x{width}!"
                  )

              all_images.append(image)
    else:
        all_images = SDXL.run(
            base_model=base_model,
            refiner=refiner,
            prompts=prompts,
            negative_prompts=negative_prompts,
            height=render_height,
            width=render_width,
            dtype=dtype,
            num_inference_steps=num_inference_steps,
            high_noise_frac=high_noise_frac,
            seeds=seeds,
          )

        if render_height != height or render_width != width:
            for i in range(len(all_images)):
                all_images[i] = all_images[i].resize((width, height)) 
         
    model = load_model(
        model_config,
        device,
        num_frames,
        num_steps,
    )
    torch.manual_seed(seed)

    for image in all_images:
        print("image type", type(image))
        image = ToTensor()(image)
        image = image * 2.0 - 1.0
        image = image.unsqueeze(0).to(device)
        H, W = image.shape[2:]
        assert image.shape[1] == 3
        F = 8
        C = 4
        shape = (num_frames, C, H // F, W // F)
        if (H, W) != (576, 1024):
            print(
                "WARNING: The conditioning frame you provided is not 576x1024. This leads to suboptimal performance as model was only trained on 576x1024. Consider increasing `cond_aug`."
            )
        if motion_bucket_id > 255:
            print(
                "WARNING: High motion bucket! This may lead to suboptimal performance."
            )

        if fps_id < 5:
            print("WARNING: Small fps value! This may lead to suboptimal performance.")

        if fps_id > 30:
            print("WARNING: Large fps value! This may lead to suboptimal performance.")

        value_dict = {}
        value_dict["motion_bucket_id"] = motion_bucket_id
        value_dict["fps_id"] = fps_id
        value_dict["cond_aug"] = cond_aug
        value_dict["cond_frames_without_noise"] = image
        value_dict["cond_frames"] = image + cond_aug * torch.randn_like(image)
        value_dict["cond_aug"] = cond_aug

        if img2img_strength is not None:
            model.sampler.discretization = Img2ImgDiscretizationWrapper(
                model.sampler.discretization,
                strength=img2img_strength,
            )

        with torch.no_grad():
            with torch.autocast(device):
                motion_video = None
                if initial_video_path is not None:
                    # only take the first num_frames
                    # load the video mp4 and convert to a tensor
                    cap = cv2.VideoCapture(initial_video_path)
                    frames = []
                    frame_count = 0
                    while cap.isOpened():
                        if frame_count >= num_frames:
                            break
                        ret, frame = cap.read()
                        if ret:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frame = ToTensor()(frame)
                            frame = frame * 2.0 - 1.0
                            frames.append(frame)
                            frame_count += 1
                        else:
                            break

                    motion_video = torch.stack(frames).to(device=device, dtype=dtype)

                    print("motion_video.shape", motion_video.shape)

                    # encode the video 
                    encoded_frames = []
                    for i in range(motion_video.shape[0]):
                        encoded_frame = model.encode_first_stage(motion_video[i].unsqueeze(0))
                        encoded_frames.append(encoded_frame)

                    motion_video = torch.concat(encoded_frames, dim=0)
                    print("motion_video.shape", motion_video.shape)


                batch, batch_uc = get_batch(
                    get_unique_embedder_keys_from_conditioner(model.conditioner),
                    value_dict,
                    [1, num_frames],
                    T=num_frames,
                    device=device,
                )
                c, uc = model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=[
                        "cond_frames",
                        "cond_frames_without_noise",
                    ],
                )

                for k in ["crossattn", "concat"]:
                    uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames)
                    uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_frames)
                    c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames)
                    c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_frames)

                randn = torch.randn(shape, device=device)
                if motion_video is not None:
                  if img2img_strength is not None:
                      sigmas = model.sampler.discretization(model.sampler.num_steps)
                      sigma = sigmas[0].to(device)

                      motion_video = motion_video + randn * append_dims(sigma, motion_video.ndim)
                      randn = motion_video / torch.sqrt(
                          1.0 + sigmas[0] ** 2.0
                      )  # Note: hardcoded to DDPM-like scaling. need to generalize later.                                          
                  else:
                    sigmas = model.sampler.discretization(1000)
                    sigma = sigmas[0].to(device)
                    print("sigma", sigma)
                    motion_video = motion_video + randn * append_dims(sigma, motion_video.ndim)
                    randn = motion_video / torch.sqrt(
                        1.0 + sigma ** 2.0
                    )  

                additional_model_inputs = {}
                additional_model_inputs["image_only_indicator"] = torch.zeros(
                    2, num_frames
                ).to(device)
                additional_model_inputs["num_video_frames"] = batch["num_video_frames"]

                def denoiser(input, sigma, c):
                    return model.denoiser(
                        model.model, input, sigma, c, **additional_model_inputs
                    )

                samples_z = model.sampler(denoiser, randn, cond=c, uc=uc)
                model.en_and_decode_n_samples_a_time = decoding_t
                samples_x = model.decode_first_stage(samples_z)
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

                image_folder = save_sampled_images(samples, output_folder)
                if interpolate:
                  interpolation_folder = image_folder + "_interpolation"

                  run_interpolate(
                      output_folder_path=interpolation_folder,
                      input_folder_path=image_folder,)

                  frames_to_video(
                      interpolation_folder, 
                      interpolation_folder + ".mp4", 
                      pattern='%07d.png',
                      frame_rate=30)  
                else:
                  frames_to_video(image_folder, image_folder + ".mp4")

def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch(keys, value_dict, N, T, device):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to(device),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames":
            batch[key] = repeat(value_dict["cond_frames"], "1 ... -> b ...", b=N[0])
        elif key == "cond_frames_without_noise":
            batch[key] = repeat(
                value_dict["cond_frames_without_noise"], "1 ... -> b ...", b=N[0]
            )
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def load_model(
    config: str,
    device: str,
    num_frames: int,
    num_steps: int,
):
    config = OmegaConf.load(config)
    if device == "cuda":
        config.model.params.conditioner_config.params.emb_models[
            0
        ].params.open_clip_embedding_config.params.init_device = device

    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.guider_config.params.num_frames = (
        num_frames
    )
    if device == "cuda":
        with torch.device(device):
            model = instantiate_from_config(config.model).to(device).eval()
    else:
        model = instantiate_from_config(config.model).to(device).eval()

    return model


if __name__ == "__main__":
    Fire(sample)
