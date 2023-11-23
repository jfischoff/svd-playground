import torch
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
from pathlib import Path

def run(base_model,
        refiner,
        vae_override=None,
        height=768,
        width=1024,
        prompts="",
        negative_prompts = "(deformediris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
        sampler_class=EulerAncestralDiscreteScheduler,
        dtype=torch.float16,
        num_inference_steps = 20,
        high_noise_frac = 0.8,  
        seeds=42,
        ):
  
  cuda_device = torch.device("cuda")

  variant = "fp16" if dtype == torch.float16 else "fp32"

  base = StableDiffusionXLPipeline.from_pretrained(
      base_model, 
      torch_dtype=dtype, 
      variant=variant, 
      use_safetensors=True,
  )
  base.to(cuda_device)
  base.scheduler = sampler_class.from_config(base.scheduler.config)

  # TODO optionally override the vae

  refiner = DiffusionPipeline.from_pretrained(
      refiner,
      text_encoder_2=base.text_encoder_2,
      vae=base.vae,
      torch_dtype=dtype,
      use_safetensors=True,
      variant=variant,
  )
  refiner.to(cuda_device)
  refiner.enable_vae_slicing()

  # turn the prompts into a list of strings
  # and the seeds into a list of generators
  # make sure the lengths match

  if isinstance(prompts, str):
    prompts = [prompts]
  if isinstance(negative_prompts, str):
    negative_prompts = [negative_prompts]

  if len(prompts) != len(negative_prompts):
    raise ValueError("Prompts and negative prompts must be the same length")

  if isinstance(seeds, int):
    seeds = [seeds] * len(prompts)
  
  # make the generators
  generators = [torch.manual_seed(seed) for seed in seeds]

  image = base(
      prompt=prompts,
      negative_prompt=negative_prompts,
      height=height,
      width=width,
      num_inference_steps=num_inference_steps,
      denoising_end=high_noise_frac,
      output_type="latent",
      generator=generators,
  ).images

  images = refiner(
      prompt=prompts,
      negative_prompt=negative_prompts,
      num_inference_steps=num_inference_steps,
      denoising_start=high_noise_frac,
      image=image,
      generator=generators,
  ).images

  del base
  del refiner

  return images

if __name__ == "__main__":
  images = run(
    base_model="/mnt/newdrive/models/stable-diffusion-xl-base-1.0/",
    refiner="/mnt/newdrive/models/stable-diffusion-xl-refiner-1.0/",
    prompts="Cute Chunky anthropomorphic Siamese cat dressed in rags walking down a rural road, mindblowing illustration by Jean-Baptiste Monge + Emily Carr + Tsubasa Nakai",
    negative_prompts="compression artifacts, blurry, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, made by children, caricature, ugly, boring, sketch, lacklustre, repetitive, cropped, (long neck), body horror, out of frame, mutilated, tiled, frame, border",
    height=1024,
    width=768,
    dtype=torch.float16,
    num_inference_steps=20,
    high_noise_frac=0.8,
    seeds=42,
  )

  # TODO save the images to a folder
  images[0].save("outputs/image.png")