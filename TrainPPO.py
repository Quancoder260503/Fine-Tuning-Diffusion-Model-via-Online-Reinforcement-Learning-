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

def collect_trajectories(
  args,
  model_pipeline,
  is_ddp,
  batch,
  calculate_reward,
  state_dict
):
  for _ in range(args.g_step):
    with torch.no_grad():
      image, latents_list, unconditional_prompt_embeds, guided_prompt_embeds, \
        log_prob_list, _ = model_pipeline.forward_collect_traj_ddim(prompt = batch, is_ddp = is_ddp)
    reward_lst = []
    text_embed_lst = []

    for i in range(len(batch)):
      reward, text_embed = calculate_reward(image[i], batch[i])
      reward_lst.append(reward)
      text_embed_lst.append(text_embed)

    reward_lst = torch.stack(reward_lst).detach().cpu()
    text_embed_lst = torch.stack(text_embed_lst).detach().cpu()

    for i in range(len(latents_list) - 1):
      state_dict["prompt"].extend(batch)
      state_dict["state"] = torch.cat((state_dict["state"], latents_list[i]))
      state_dict["next_state"] = torch.cat((state_dict["next_state"], latents_list[i + 1]))
      state_dict["timestep"] = torch.cat((state_dict["timestep"], torch.LongTensor([i] * len(batch))))
      state_dict["final_reward"] = torch.cat((state_dict["final_reward"], reward_lst))
      state_dict["unconditional_prompt_embeds"] = torch.cat((
       state_dict["unconditional_prompt_embeds"],
       unconditional_prompt_embeds
      ))
      state_dict["guided_prompt_embeds"] = torch.cat((
       state_dict["guided_prompt_embeds"],
       guided_prompt_embeds
      ))
      state_dict["text_embed"] = torch.cat((state_dict["text_embed"], text_embed_lst))
      state_dict["log_prob"] = torch.cat((state_dict["log_prob"], log_prob_list[i]))

    print(f"final_reward mean/std this batch: {state_dict['final_reward'].mean().item(), state_dict['final_reward'].std().item()}")
    del (
      image,
      latents_list,
      unconditional_prompt_embeds,
      guided_prompt_embeds,
      reward_lst,
      text_embed_lst,
      log_prob_list,
      reward,
      text_embed
    )
    torch.cuda.empty_cache()

def train_value_func(
  value_approximator_fn,
  state_dict,
  accelerator,
  args,
):
  indices = get_random_indices(state_dict['state'].shape[0], args.value_batch_size)
  batch_state = state_dict["state"][indices]
  batch_timestep = state_dict["timestep"][indices]
  batch_final_reward = state_dict["final_reward"][indices]
  batch_text_embed = state_dict["text_embed"][indices]

  pred_value = value_approximator_fn(
    batch_state.cuda().detach(),
    batch_text_embed.cuda().detach(),
    batch_timestep.cuda().detach()
  )
  batch_final_reward = batch_final_reward.cuda().float()
  value_loss = F.mse_loss(pred_value.float().reshape([args.value_batch_size, 1]), \
    batch_final_reward.cuda().detach().reshape([args.value_batch_size, 1]))

  accelerator.backward(value_loss / args.value_gradient_step)

  del pred_value
  del batch_state
  del batch_timestep
  del batch_final_reward
  del batch_text_embed

  return (value_loss.item() / args.value_gradient_step)


@dataclasses.dataclass(frozen = False)
class TrainPolicyFunctionData:
  total_policy_loss : float = 0.
  total_ratio : float = 0.
  total_kl_divergence : float = 0.
  total_gradient_norm : float = 0.

def train_policy_func(
  args,
  state_dict,
  model_pipeline,
  unet_copy,
  is_ddp,
  count,
  policy_steps,
  accelerator,
  tpfdata,
  value_function, 
):
  with torch.no_grad():
    indices = get_random_indices(state_dict['state'].shape[0], args.policy_batch_size)
    batch_state = state_dict["state"][indices]
    batch_next_state = state_dict["next_state"][indices]
    batch_timestep = state_dict["timestep"][indices]
    batch_final_reward = state_dict["final_reward"][indices]
    batch_guided_prompt_embeds = state_dict["guided_prompt_embeds"][indices]
    batch_unconditional_prompt_embeds = state_dict["unconditional_prompt_embeds"][indices]

    batch_prompt_embeds = torch.cat([batch_unconditional_prompt_embeds, batch_guided_prompt_embeds])
    batch_text_embed = state_dict["text_embed"][indices]
    batch_log_prob = state_dict["log_prob"][indices]


  log_prob, kl_regularizer = model_pipeline.forward_calculate_log_prob(
    prompt_embeds = batch_prompt_embeds.cuda(),
    latents = batch_state.cuda(),
    next_latents = batch_next_state.cuda(),
    time = batch_timestep.cuda(),
    unet_copy = unet_copy,
    is_ddp = is_ddp
  )

  with torch.no_grad():
    if args.value_flag == 1:
      approximated_value = value_function(
        batch_state.cuda(),
        batch_text_embed.cuda(),
        batch_timestep.cuda()
      ).reshape([args.policy_batch_size, 1])
      advantages = batch_final_reward.cuda().reshape([args.policy_batch_size, 1]) - approximated_value
    else:
      advantages = batch_final_reward.cuda().reshape([args.policy_batch_size, 1])
  ratio = torch.exp(log_prob - batch_log_prob.cuda())
  ratio = torch.clamp(ratio, 1.0 - args.ratio_clip, 1.0 + args.ratio_clip)
  loss = -args.reward_weight * advantages.detach().float() * ratio.float().reshape([args.policy_batch_size, 1])
  loss = loss.mean()
 # print(loss, ratio, 1.0 - args.ratio_clip, 1.0 + args.ratio_clip)
  if count > args.kl_warmup:
    loss += args.kl_weight * kl_regularizer.mean()

  accelerator.backward(loss / args.gradient_accumulation_steps)

  tpfdata.total_ratio += ratio.mean().item() / policy_steps
  tpfdata.total_kl_divergence += kl_regularizer.mean().item() / policy_steps
  tpfdata.total_policy_loss += loss.item() / policy_steps


def main():
  args = parse_args()
  update_output_dir(args, algo = 'PPO')
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

  value_function_raw = CombineValueApproximator(
    num_diffusion_steps = 50,
    img_shape = (4, 64, 64),
    text_dim = 768,
    layer_dims = [256, 256, 256],
    activation = 'gelu'
  )
  value_optimizer = torch.optim.AdamW(
    params = value_function_raw.parameters(),
    lr = args.value_lr,
  )

  value_function, value_optimizer = accelerator.prepare(
    value_function_raw, value_optimizer
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
      path = os.path.basename(args.resume_from_checkpoint)

    if path is None:
      accelerator.print(
        f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
      )
      args.resume_from_checkpoint = None
    else:
      accelerator.print(f"Resume from checkpoint {path}")
      accelerator.load_state(os.path.join(args.output_dir, path))
      global_step = int(path.split("-")[1])

  logger.info("=" * 40, 'Start PPO Training', '=' * 40)
  logger.info(f" Num Epochs = {args.num_train_epochs}")
  logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
  logger.info(f"Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
  logger.info(f"Gradient Accumulation steps = {args.gradient_accumulation_steps}")
  logger.info(f"Total optimization steps = {args.max_train_steps // args.policy_step}")

  model_pipeline.scheduler = ExtendedDDIMScheduler.from_config(model_pipeline.scheduler.config)
  model_pipeline.to("cuda")

  progress_bar = tqdm(range(global_step, args.max_train_steps), disable = not accelerator.is_main_process)

  progress_bar.set_description("Steps")


  state_dict = {}
  state_dict["prompt"] = []
  state_dict["state"] = torch.FloatTensor().to(weight_dtype).cpu()
  state_dict["next_state"] = torch.FloatTensor().to(weight_dtype).cpu()
  state_dict["timestep"] = torch.LongTensor().cpu()
  state_dict["final_reward"] = torch.FloatTensor().to(weight_dtype).cpu()
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
    collect_trajectories(args, model_pipeline, is_ddp, batch, calculate_reward, state_dict)
    trim_buffer(buffer_size, state_dict)

    # train the critics
    if args.value_flag == 1:
      total_value_loss = 0.
      value_optimizer.zero_grad()
      for value_step in range(args.value_step):
        if value_step + 1 < args.value_step:
          with accelerator.no_sync(value_function):
            total_value_loss += train_value_func(
              value_function,
              state_dict,
              accelerator,
              args,
            )
        else:
          total_value_loss += train_value_func(
            value_function,
            state_dict,
            accelerator,
            args,
          )

      value_optimizer.step()
      if accelerator.is_main_process:
        print("value_loss", total_value_loss)
        accelerator.log({"value_loss": total_value_loss}, step = count)
      del total_value_loss
      torch.cuda.empty_cache()

    # train the actor
    tpfdata = TrainPolicyFunctionData()
    norm = torch.tensor(0.0)
    for _ in range(args.policy_step):
      optimizer.zero_grad()
      for accum_step in range(int(args.gradient_accumulation_steps)):
        if accum_step + 1 < int(args.gradient_accumulation_steps):
          with accelerator.no_sync(unet):
            train_policy_func(
              args = args,
              state_dict = state_dict,
              model_pipeline = model_pipeline,
              unet_copy = unet_copy,
              is_ddp = is_ddp,
              count = count,
              policy_steps = policy_steps,
              accelerator = accelerator,
              tpfdata = tpfdata,
              value_function = value_function,
            )
        else:
          train_policy_func(
            args = args,
            state_dict = state_dict,
            model_pipeline = model_pipeline,
            unet_copy = unet_copy,
            is_ddp = is_ddp,
            count = count,
            policy_steps = policy_steps,
            accelerator = accelerator,
            tpfdata = tpfdata,
            value_function = value_function,
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