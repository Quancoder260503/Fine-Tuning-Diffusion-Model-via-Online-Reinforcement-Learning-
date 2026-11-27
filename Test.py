import argparse
import os
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from torchvision.transforms.functional import to_tensor
from torchvision.utils import make_grid, save_image


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--prompt', type = str, required = True)
  parser.add_argument('--model_path', type = str, default = None,
                      help = 'folder with the LoRA weights; ignored with --base')
  parser.add_argument('--base', action = 'store_true', help = 'sample from the original model, no LoRA')
  parser.add_argument('--out_dir', type = str, default = 'samples')
  parser.add_argument('--num_images', type = int, default = 10)
  parser.add_argument('--nrow', type = int, default = 5, help = 'images per row in the grid')
  parser.add_argument('--seed', type = int, default = 0)
  parser.add_argument('--steps', type = int, default = 50, help = 'DDIM steps; match training')
  parser.add_argument('--guidance_scale', type = float, default = 7.5, help = 'cfg scale; match training')
  parser.add_argument('--eta', type = float, default = 1.0)
  parser.add_argument('--batch_size', type = int, default = 5)
  return parser.parse_args()


def main(args):
  pipe = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', torch_dtype = torch.float16)
  pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
  if not args.base :
    pipe.load_lora_weights(args.model_path)          
  pipe.to('cuda')
  pipe.set_progress_bar_config(disable = True)

  os.makedirs(args.out_dir, exist_ok = True)

  seeds = [args.seed + i for i in range(args.num_images)]

  samples = []
  for start in range(0, args.num_images, args.batch_size):
    batch_seeds = seeds[start : start + args.batch_size]
    generators = [torch.Generator('cuda').manual_seed(s) for s in batch_seeds]
    images = pipe(
      prompt = [args.prompt] * len(batch_seeds),
      generator = generators,
      num_inference_steps = args.steps,
      guidance_scale = args.guidance_scale,
      eta = args.eta,
    ).images
    for s, image in zip(batch_seeds, images):
      image.save(os.path.join(args.out_dir, f'seed_{s:04d}.png'))
      samples.append(to_tensor(image))              
    print(f'generated {len(samples)} / {args.num_images}')

  grid = make_grid(torch.stack(samples), nrow = args.nrow, padding = 4, pad_value = 0.0)
  save_image(grid, os.path.join(args.out_dir, 'grid.png'))
  print(f'saved {len(samples)} images and grid.png to {args.out_dir}')


if __name__ == '__main__':
  main(parse_args())