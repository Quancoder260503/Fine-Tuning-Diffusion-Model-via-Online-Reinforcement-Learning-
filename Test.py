from argparse import ArgumentParser  
import random
from diffusers import DDIMScheduler
from diffusers import StableDiffusionPipeline  
import numpy as np
import torch
import os 


def main(args):
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed(args.seed)
  #torch.backends.cudnn.deterministic = True
 # torch.backends.cudnn.benchmark = False
  np.random.seed(args.seed)
  random.seed(args.seed)
  model_path = args.model_path
  pipe = StableDiffusionPipeline.from_pretrained( "runwayml/stable-diffusion-v1-5")
  pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
  pipe.load_lora_weights(model_path)
  sample_key = None

  for name, param in pipe.unet.named_parameters():
      if "to_q" in name and "weight" in name:
        sample_key = name
        before = param.detach().clone()
        break

  pipe.load_lora_weights(model_path)
  
  # a = dict(pipe.unet.named_parameters())
  # name0 = "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_B.default_0.weight"
  # name1 = "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_B.default_1.weight"
  # print(torch.equal(a[name0], a[name1]))

  # # check 1: are there any LoRA parameters in the unet at all?
  # lora_params = [n for n, _ in pipe.unet.named_parameters() if "lora" in n.lower()]
  # print(f"Number of LoRA parameters found: {len(lora_params)}")
  # if lora_params:
  #     print("Example LoRA param names:", lora_params[:5])
  # else:
  #     print("WARNING: no LoRA parameters found — weights likely did not load")

  # if hasattr(pipe.unet, "peft_config"):
  #     print("Active adapter config:", pipe.unet.peft_config)
  # else:
  #     print("WARNING: unet has no peft_config — adapter not attached")

  # for name, param in pipe.unet.named_parameters():
  #     if "lora_A" in name or "lora_B" in name:
  #         print(name, "nonzero:", torch.any(param != 0).item())
  pipe.to("cuda")
  prompt_list = [args.prompt for i in range(10)]
  for i in range(10):
    images = pipe(prompt = prompt_list[i], eta = 1.0).images
    image = images[0]
    save_dir = os.path.dirname(model_path)
    image.save(os.path.join(save_dir, "image{}.png".format(i)))


if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--prompt", default = "A green colored rabbit.")
  parser.add_argument("--model-path", default = "online_model/img_reward_0/pre_train/single_prompt/A_green_colored_rabbit./p_lr1e-05_s5_b4_g12_l4_kl0.01_re10.0_v_lr0.0001_b256_s5/save_1100/pytorch_lora_weights.safetensors")
  parser.add_argument("--seed", default = 0, type = int)
  main(parser.parse_args())