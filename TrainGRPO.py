import dataclasses 
import functools 
import json 
import logging
import os 
import pickle 
import random 
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration 
import datasets
from datasets import load_dataset
import diffusers
from diffusers import UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from peft import LoraConfig
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils.import_utils import is_xformers_available
from packaging import version
from ModelPipeline import *
from RewardModel import *
from SchedulingDDIM import *
import ImageReward as imagereward
import Utils 
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from tqdm.auto import tqdm
import transformers
from transformers import CLIPModel, CLIPProcessor
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate.logging import get_logger
from huggingface_hub import login
from diffusers.image_processor import VaeImageProcessor
from Parser import * 
from Utils import *

logger = get_logger(__name__, log_level = "INFO")

"""
Group Reward Policy Optimization (GRPO) for text-to-image generation with Stable Diffusion. The idea well, 
is exactly what GRPO does, for each prompt, we generate multiple samples (images) and compute the reward 
for each sample. Then we standardize the rewards within the group of samples for each prompt, and use these 
standardized rewards to update the diffusion model (as diffusion process can be modelled as a Markov Decision Process, 
so Reinforcement Learning approach comes quite naturally since the transition model are approximated and reward function 
is unknown). 
"""

def grpo_collect_trajectories(
  args,
  model_pipeline,
  is_ddp,
  batch,
  calculate_reward,
  state_dict
):
  # prompt
  num_samples_in_batch = len(batch)
  new_batch = []
  for prompt in batch:
    new_batch.extend([prompt] * args.grpo_samples_per_group) 
  batch = new_batch

  with torch.no_grad():
    image, latents_list, unconditional_prompt_embeds, guided_prompt_embeds, \
      log_prob_list, _ = model_pipeline.forward_collect_traj_ddim(prompt = batch, is_ddp = is_ddp)
  
  
  reward_lst = []
  text_embed_lst = []
  advantage_lst = [] 
    
  for i in range(num_samples_in_batch):
    curr_reward_group = []
    for j in range(args.grpo_samples_per_group):
      index = i * args.grpo_samples_per_group + j
      reward, text_embed = calculate_reward(image[index], batch[index]) 

      reward = reward.detach().float().cpu()
      text_embed = text_embed.detach().cpu()
      curr_reward_group.append(reward)
      text_embed_lst.append(text_embed)

    curr_reward_group = torch.stack(curr_reward_group)
    curr_group_mean = curr_reward_group.mean()
    curr_group_std = curr_reward_group.std()
    
    reward_lst.extend(curr_reward_group)

    curr_advantage_group = (curr_reward_group - curr_group_mean) / (curr_group_std + args.denom_eps) # standardize the reward within the group of samples for each prompt  
    advantage_lst.extend(curr_advantage_group)


  reward_lst = torch.stack(reward_lst).detach().cpu()
  advantage_lst = torch.stack(advantage_lst).detach().cpu()
  text_embed_lst = torch.stack(text_embed_lst).detach().cpu()

  state_chunks = []
  next_state_chunks = []
  timestep_chunks = []
  final_reward_chunks = []  
  uncond_embed_chunks = []
  guided_embed_chunks = []
  text_embed_chunks = []
  log_prob_chunks = []
  prompt_chunks = []
  final_advantage_chunks = [] 
  
   
  for i in range(len(latents_list) - 1):
    prompt_chunks.extend(batch)
    state_chunks.append(latents_list[i])
    next_state_chunks.append(latents_list[i + 1])
    timestep_chunks.append(torch.LongTensor([i] * len(batch)))
    final_reward_chunks.append(reward_lst)
    final_advantage_chunks.append(advantage_lst)
    uncond_embed_chunks.append(unconditional_prompt_embeds)
    guided_embed_chunks.append(guided_prompt_embeds)
    text_embed_chunks.append(text_embed_lst)
    log_prob_chunks.append(log_prob_list[i])

  state_dict["prompt"].extend(prompt_chunks)
  state_dict["state"] = torch.cat([state_dict["state"]] + state_chunks)
  state_dict["next_state"] = torch.cat([state_dict["next_state"]] + next_state_chunks)
  state_dict["timestep"] = torch.cat([state_dict["timestep"]] + timestep_chunks)
  state_dict["final_reward"] = torch.cat([state_dict["final_reward"]] + final_reward_chunks)
  state_dict["final_advantage"] = torch.cat([state_dict["final_advantage"]] + final_advantage_chunks)
  state_dict["unconditional_prompt_embeds"] = torch.cat([state_dict["unconditional_prompt_embeds"]] + uncond_embed_chunks)
  state_dict["guided_prompt_embeds"] = torch.cat([state_dict["guided_prompt_embeds"]] + guided_embed_chunks)
  state_dict["text_embed"] = torch.cat([state_dict["text_embed"]] + text_embed_chunks)
  state_dict["log_prob"] = torch.cat([state_dict["log_prob"]] + log_prob_chunks)


  print(f"\nfinal_reward mean/std this batch: {state_dict['final_reward'].mean().item(), state_dict['final_reward'].std().item()}")

  del (
    image,
    latents_list,
    unconditional_prompt_embeds,
    guided_prompt_embeds,
    reward_lst,
    advantage_lst, 
    text_embed_lst,
    log_prob_list,
  )
  torch.cuda.empty_cache()

def clear_buffer(state_dict):
  state_dict["prompt"] = []
  state_dict["state"] = torch.FloatTensor().to(state_dict["state"].dtype).cpu()
  state_dict["next_state"] = torch.FloatTensor().to(state_dict["next_state"].dtype).cpu()
  state_dict["timestep"] = torch.LongTensor().cpu()
  state_dict["final_reward"] = torch.FloatTensor().to(state_dict["final_reward"].dtype).cpu()
  state_dict["final_advantage"] = torch.FloatTensor().to(state_dict["final_advantage"].dtype).cpu()
  state_dict["unconditional_prompt_embeds"] = torch.FloatTensor().to(state_dict["unconditional_prompt_embeds"].dtype).cpu()
  state_dict["guided_prompt_embeds"] = torch.FloatTensor().to(state_dict["guided_prompt_embeds"].dtype).cpu()
  state_dict["text_embed"] = torch.FloatTensor().to(state_dict["text_embed"].dtype).cpu()
  state_dict["log_prob"] = torch.FloatTensor().to(state_dict["log_prob"].dtype).cpu()


@dataclasses.dataclass(frozen = False)
class TrainPolicyFunctionData:
  total_policy_loss : float = 0.
  total_ratio : float = 0.
  total_kl_divergence : float = 0.
  total_gradient_norm : float = 0.

def train_policy_func(
  args,
  start_idx, 
  end_idx, 
  state_dict,
  model_pipeline,
  unet_copy,
  is_ddp,
  count,
  policy_steps,
  accelerator,
  tpfdata,
):

  
  with torch.no_grad():
    batch_state = state_dict["state"][start_idx:end_idx]
    batch_next_state = state_dict["next_state"][start_idx:end_idx]
    batch_timestep = state_dict["timestep"][start_idx:end_idx]
    batch_final_reward = state_dict["final_reward"][start_idx:end_idx]
    batch_final_advantage = state_dict["final_advantage"][start_idx:end_idx]
    batch_guided_prompt_embeds = state_dict["guided_prompt_embeds"][start_idx:end_idx]
    batch_unconditional_prompt_embeds = state_dict["unconditional_prompt_embeds"][start_idx:end_idx]

    batch_prompt_embeds = torch.cat([batch_unconditional_prompt_embeds, batch_guided_prompt_embeds])
    batch_text_embed = state_dict["text_embed"][start_idx:end_idx]
    batch_log_prob = state_dict["log_prob"][start_idx:end_idx]


  log_prob, kl_regularizer = model_pipeline.forward_calculate_log_prob(
    prompt_embeds = batch_prompt_embeds.cuda(),
    latents = batch_state.cuda(),
    next_latents = batch_next_state.cuda(),
    time = batch_timestep.cuda(),
    unet_copy = unet_copy,
    is_ddp = is_ddp
  )

  advantages = batch_final_advantage.cuda().reshape([args.policy_batch_size, 1])

  ratio = torch.exp(log_prob - batch_log_prob.cuda())
  ratio = ratio.float().reshape([args.policy_batch_size, 1])
  adv = advantages.detach().float()

  unclipped_loss = -adv * ratio
  clipped_loss = -adv * torch.clamp(ratio, 1.0 - args.ratio_clip, 1.0 + args.ratio_clip)

  loss = args.reward_weight * torch.maximum(unclipped_loss, clipped_loss)
  loss = loss.mean()
  
  if count > args.kl_warmup:
   loss += args.kl_weight * kl_regularizer.mean()
  
  
  accelerator.backward(loss / args.gradient_accumulation_steps)

  tpfdata.total_ratio += ratio.mean().item() / policy_steps
  tpfdata.total_kl_divergence += kl_regularizer.mean().item() / policy_steps
  tpfdata.total_policy_loss += loss.item() / policy_steps


def main():
  args = parse_args()
  update_output_dir(args, algo = 'GRPO')
  logging_dir = os.path.join(args.output_dir, args.logging_dir)
  accelerator_proj_config = ProjectConfiguration(
    logging_dir = logging_dir,
    total_limit = args.checkpoints_total_limit,
  )

  accelerator = Accelerator(
    mixed_precision = args.mixed_precision,
    log_with = args.report_to,
    project_config = accelerator_proj_config,
  )

  logging.basicConfig(
    format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt = "%m/%d/%Y %H:%M:%S",
    level = logging.INFO,
  )

  logger.info(accelerator.state, main_process_only = False)
  if accelerator.is_local_main_process:
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_warning()
    diffusers.utils.logging.set_verbosity_warning()
  else:
    datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()
    diffusers.utils.logging.set_verbosity_error()

  if args.seed is not None:
    set_seed(args.seed)

  if accelerator.is_main_process:
    os.makedirs(args.output_dir, exist_ok = True)

  tokenizer = CLIPTokenizer.from_pretrained(
    args.pretrained_model_path,
    subfolder = "tokenizer",
    revision = args.revision
  )

  text_encoder = CLIPTextModel.from_pretrained(
    args.pretrained_model_path,
    subfolder = "text_encoder",
    revision = args.revision
  )

  weight_dtype = torch.float32
  if accelerator.mixed_precision == "fp16":
    weight_dtype = torch.float16
  elif accelerator.mixed_precision == "bf16":
    weight_dtype = torch.bfloat16
  
  # Loading reward model
  reward_clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
  reward_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
  reward_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

  if args.reward_flag == 0:
    reward_model = imagereward.load("ImageReward-v1.0")
    reward_model.requires_grad_(False)
    reward_model.to(accelerator.device, dtype = weight_dtype)
  else:
    reward_model = pickle.load(open(args.reward_model_path, "rb"))["reward"]
    reward_model.requires_grad_(False)
    reward_model.to(accelerator.device, dtype = weight_dtype)

  # Gotta frozen the reward model 
  reward_clip_model.requires_grad_(False)
  
  if args.sft_initialization == 0:
    model_pipeline = ExtendStableDiffusionPipeline.from_pretrained(
      args.pretrained_model_path,
      torch_dtype = weight_dtype
    )
    unet = UNet2DConditionModel.from_pretrained(
      args.pretrained_model_path,
      subfolder = "unet",
      revision = args.non_ema_revision
    )
  else:
    model_pipeline = ExtendStableDiffusionPipeline.from_pretrained(
      args.pretrained_model_path,
      torch_dtype = weight_dtype
    )
    unet = UNet2DConditionModel.from_pretrained(
      args.sft_path,
      subfolder = "unet",
      revision = args.non_ema_revision
    )

  # Loading the vae, I think one of my research direction would be see if we can replace this source of inductive bias with a 
  # non-training one while still maintaining the same level of performance.
  vae = model_pipeline.vae
  unet.requires_grad_(False)
  unet.eval()
  text_encoder = model_pipeline.text_encoder

  vae.requires_grad_(False)
  text_encoder.requires_grad_(False)

  unet_copy = UNet2DConditionModel.from_pretrained(
    args.pretrained_model_path,
    subfolder = "unet",
    revision = args.non_ema_revision,
  )

  unet_copy.requires_grad_(False)
  text_encoder.to(accelerator.device, dtype = weight_dtype)
  vae.to(accelerator.device, dtype = weight_dtype)
  unet.to(accelerator.device, dtype = weight_dtype)
  unet_copy.to(accelerator.device, dtype = weight_dtype)
  
  # Use EMA model for evaluation if specified
  if args.use_ema:
    ema_unet = UNet2DConditionModel.from_pretrained(
      args.sft_path,
      subfolder = "unet",
      revision = args.non_ema_revision
    )
    ema_unet = EMAModel(
      ema_unet.parameters(),
      model_cls = UNet2DConditionModel,
      model_config = ema_unet.config,
    )

  if args.enable_xformers_memory_efficient_attention:
    if is_xformers_available():
      import xformers
      xformers_version = version.parse(xformers.__version__)
      if xformers_version == version.parse("0.0.16"):
        logger.warning(
          "xFormers 0.0.16 cannot be used for training in some GPUs. If you"
          " observe problems during training, please update xFormers to at"
          " least 0.0.17. See"
          " https://huggingface.co/docs/diffusers/main/en/optimization/xformers"
          " for more details."
        )
      unet.enable_xformers_memory_efficient_attention()
    else:
      raise ValueError(
        "xformers is not available. Make sure it is installed correctly"
      )


  lora_config = LoraConfig(
    r = args.lora_rank,
    lora_alpha = args.lora_rank,
    target_modules = ["to_q", "to_v", "to_k", "to_out.0"],
    lora_dropout = 0.0,
    bias = "none",
  )
  unet.add_adapter(lora_config)
  for param in unet.parameters():
    if param.requires_grad:
      param.data = param.data.to(torch.float32)
  lora_layers = [p for p in unet.parameters() if p.requires_grad]



  if args.gradient_checkpointing:
    unet.enable_gradient_checkpointing()

  if args.scale_lr:
    args.learning_rate = args.learning_rate * args.gradient_accumulation_steps \
      * args.train_batch_size * accelerator.num_processes

  if args.use_8bit_adam:
    try:
      import bitsandbytes
    except Exception as exc:
      raise ImportError(
        "Please install bitsandbytes to use 8-bit Adam. You can do so by"
        " running `pip install bitsandbytes`"
      ) from exc

    optimizer = bitsandbytes.optim.AdamW8bit(
      params = lora_layers,
      lr = args.learning_rate,
      betas = (args.adam_beta1, args.adam_beta2),
      weight_decay = args.adam_weight_decay,
      eps = args.adam_epsilon,
    )
  else:
    optimizer = torch.optim.AdamW(
      params = lora_layers,
      lr = args.learning_rate,
      betas = (args.adam_beta1, args.adam_beta2),
      weight_decay = args.adam_weight_decay,
      eps = args.adam_epsilon,
    )


  if args.dataset_name is not None:
    load_dataset(
      args.dataset_name,
      args.dataset_config_name,
      cache_dir = args.cache_dir,
    )
  else:
    with open(args.prompt_path) as json_file:
      prompt_dict = json.load(json_file)

    if args.prompt_category != "all":
      prompt_category = [e for e in args.prompt_category.split(",")]

    prompt_list = []
    for prompt in prompt_dict:
      category = prompt_dict[prompt]["category"]
      if args.prompt_category != "all":
        if category in prompt_category:
          prompt_list.append(prompt)
      else:
        prompt_list.append(prompt)

  def custom_data_iter(data, batch_size):
    random.shuffle(data)
    for i in range(0, len(data), batch_size):
      batch = data[i : i + batch_size]
      yield batch

  data_iter = custom_data_iter(prompt_list, batch_size = args.g_batch_size)
  data_iter = accelerator.prepare(data_iter)
  lr_scheduler = get_scheduler(
    name = args.lr_scheduler,
    optimizer = optimizer,
    num_warmup_steps = args.lr_warmup_steps * args.gradient_accumulation_steps,
    num_training_steps = args.max_train_steps * args.gradient_accumulation_steps,
  )


  if args.multi_gpu:
    unet, optimizer, lr_scheduler = accelerator.prepare(
      unet, optimizer, lr_scheduler
    )
  else:
    lora_layers, optimizer, lr_scheduler = accelerator.prepare(
      lora_layers, optimizer, lr_scheduler
    )

  if args.use_ema:
    ema_unet.to(accelerator.device)

  reward_clip_model.to(accelerator.device, dtype = weight_dtype)

  if accelerator.is_main_process:
    accelerator.init_trackers("text2image-fine-tuning", config = vars(args))

  total_batch_size = args.train_batch_size * accelerator.num_processes *\
    args.gradient_accumulation_steps

  global_step = 0

  if args.resume_from_checkpoint:
    if args.resume_from_checkpoint == "latest":
      dirs = os.listdir(args.output_dir)
      dirs = [d for d in dirs if d.startswith("checkpoint")]
      dirs = sorted(dirs, key = lambda x : int(x.split("-")[1]))
      path = dirs[-1] if len(dirs) > 0 else None
    else:
      path = args.resume_from_checkpoint

    if path is None:
      accelerator.print(
        f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
      )
      args.resume_from_checkpoint = None
    else:
      accelerator.print(f"Resume from checkpoint {path}")
      accelerator.load_state(path)
      global_step = int(path.split("-")[1])

  logger.info("============================= Start GRPO Training =============================")
  logger.info(f" Num Epochs = {args.num_train_epochs}")
  logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
  logger.info(f"Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
  logger.info(f"Gradient Accumulation steps = {args.gradient_accumulation_steps}")
  logger.info(f"Total optimization steps = {args.max_train_steps // args.policy_step}")

  model_pipeline.scheduler = ExtendedDDIMScheduler.from_config(model_pipeline.scheduler.config)
  model_pipeline.to("cuda")

  progress_bar = tqdm(range(global_step, args.max_train_steps), disable = not accelerator.is_main_process)

  progress_bar.set_description("Steps")

  # setting up the buffer to store the trajectories for each prompt 
  state_dict = {}
  state_dict["prompt"] = []
  state_dict["state"] = torch.FloatTensor().to(weight_dtype).cpu()
  state_dict["next_state"] = torch.FloatTensor().to(weight_dtype).cpu()
  state_dict["timestep"] = torch.LongTensor().cpu()
  state_dict["final_reward"] = torch.FloatTensor().to(weight_dtype).cpu()
  state_dict["final_advantage"] = torch.FloatTensor().to(weight_dtype).cpu()
  state_dict["unconditional_prompt_embeds"] = torch.FloatTensor().to(weight_dtype).cpu()
  state_dict["guided_prompt_embeds"] = torch.FloatTensor().to(weight_dtype).cpu()
  state_dict["text_embed"] = torch.FloatTensor().to(weight_dtype).cpu()
  state_dict["log_prob"] = torch.FloatTensor().to(weight_dtype).cpu()
 
  if args.reward_flag == 0:
    calculate_reward = functools.partial(
      calculate_reward_ir,
      model_pipeline,
      args,
      reward_tokenizer,
      tokenizer,
      weight_dtype,
      reward_clip_model,
      reward_model,
    )
  else:
    calculate_reward = functools.partial(
      compute_reward_custom,
      model_pipeline,
      args,
      reward_tokenizer,
      tokenizer,
      weight_dtype,
      reward_clip_model,
      reward_processor,
      reward_model,
    )

  count = 0
  buffer_size = args.buffer_size
  policy_steps = args.gradient_accumulation_steps * args.policy_step

  data_iter_loader = iter(data_iter)
  is_ddp = isinstance(unet, DistributedDataParallel)
  model_pipeline.unet = unet
  print("model is parallel:", is_ddp)


  for count in range(0, args.max_train_steps // args.policy_step):
    unet.eval()
    batch, data_iter_loader = get_batch(
      data_iter_loader = data_iter_loader,
      data_iterator = custom_data_iter,
      prompt_list = prompt_list,
      args = args,
      accelerator = accelerator,
    )
    clear_buffer(state_dict)
    grpo_collect_trajectories(args, model_pipeline, is_ddp, batch, calculate_reward, state_dict)


    # train the actor
    tpfdata = TrainPolicyFunctionData()
    norm = torch.tensor(0.0)
    start_idx = 0
    for _ in range(args.policy_step):
      optimizer.zero_grad()
      for accum_step in range(int(args.gradient_accumulation_steps)):
        train_policy_func(
          args = args,
          start_idx = start_idx, 
          end_idx = start_idx + args.policy_batch_size,
          state_dict = state_dict,
          model_pipeline = model_pipeline,
          unet_copy = unet_copy,
          is_ddp = is_ddp,
          count = count,
          policy_steps = policy_steps,
          accelerator = accelerator,
          tpfdata = tpfdata,
        )
      if accelerator.sync_gradients:
        norm = accelerator.clip_grad_norm_(unet.parameters(), args.clip_norm)

      tpfdata.total_gradient_norm += norm.item() / args.policy_step
      optimizer.step()
      lr_scheduler.step()
      if accelerator.is_main_process:
        print(f"count: [{count} / {args.max_train_steps // args.policy_step}]")
        print("train_reward", torch.mean(state_dict["final_reward"]).item())
        accelerator.log({"train_reward": torch.mean(state_dict["final_reward"]).item()}, step = count)
        print("grad norm", tpfdata.total_gradient_norm, "ratio", tpfdata.total_ratio)
        print("kl", tpfdata.total_kl_divergence, "policy_loss", tpfdata.total_policy_loss)
        accelerator.log({"grad norm": tpfdata.total_gradient_norm}, step = count)
        accelerator.log({"ratio": tpfdata.total_ratio}, step = count)
        accelerator.log({"kl": tpfdata.total_kl_divergence}, step = count)
        accelerator.log({"policy_loss": tpfdata.total_policy_loss}, step = count)
      torch.cuda.empty_cache()
      start_idx += args.policy_batch_size

    if accelerator.sync_gradients:
      global_step += 1
      if global_step % args.checkpointing_steps == 0:
        if accelerator.is_main_process:
          save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
          accelerator.save_state(output_dir = save_path)
          logger.info(f"Saved state to {save_path}")
      print("global_step", global_step)

    if count % args.save_interval == 0:
      accelerator.wait_for_everyone()
      if accelerator.is_main_process:
        save_model(args, count, is_ddp, accelerator, unet)

  accelerator.end_training()

if __name__ == "__main__":
  main()