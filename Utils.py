from ImageReward import ImageReward
import torch
import numpy as np
import os 
import copy 


def image_reward_get_reward(model, pil_image, prompt, weight_dtype):
  """Gets rewards using ImageReward model."""
  image = model.preprocess(pil_image).unsqueeze(0).to(weight_dtype).to(model.device)

  image_embeds = model.blip.visual_encoder(image)
  image_atts = torch.ones(image_embeds.size()[:-1], dtype = torch.long).to(model.device)

  text_input = model.blip.tokenizer(prompt, padding = "max_length", truncation = True, max_length = 35, return_tensors = "pt").to(model.device)
  text_output = model.blip.text_encoder(
    text_input.input_ids,
    attention_mask = text_input.attention_mask,
    encoder_hidden_states = image_embeds,
    encoder_attention_mask = image_atts,
    return_dict = True,
  )
  txt_features = text_output.last_hidden_state[:, 0, :]
  rewards = model.mlp(txt_features)
  rewards = (rewards - model.mean) / model.std
  return rewards, txt_features


def get_random_indices(num_indices, sample_size): 
   return np.random.choice(num_indices, size = sample_size, replace = False)

def update_output_dir(args, algo = 'PPO'):
    args.output_dir = algo + args.output_dir
    if args.single_prompt_flag == 1: 
       data_log = "single_prompt/" + args.single_prompt.replace(" ", "_") + "/"
    else : 
       data_log = args.prompt_path.split("/")[-2] + "_" 
       data_log = args.prompt_category + "/"
    
    learning_log = "p_lr" + str(args.learning_rate) + "_s" + str(args.policy_step) 
    learning_log += "_b" + str(args.policy_batch_size) + "_g" + str(args.gradient_accumulation_steps) 
    learning_log += "_l" + str(args.lora_rank) 
    
    coeff_log = "_kl" + str(args.kl_weight) + "_re" + str(args.reward_weight) 
    if args.kl_warmup > 0 : 
      coeff_log += "_klw" + str(args.kl_warmup) 
    if args.sft_initialization == 0:
        start_log = "/pre_train/"
    else:
        start_log = "/sft/"
    
    if args.reward_flag == 0:
      args.output_dir += f"/img_reward_{args.reward_filter}/"
    else:
      args.output_dir += f"/prev_reward_{args.reward_filter}/"
    
    args.output_dir += start_log + data_log + "/" + learning_log + coeff_log
    if args.value_flag == 1:
      value_log = "_v_lr" + str(args.value_lr) + "_b" + str(args.value_batch_size)
      value_log += "_s" + str(args.value_step)
    args.output_dir += value_log


def compute_reward_custom(
  data_pipeline,
  args,
  reward_tokenizer,
  tokenizer,
  weight_dtype,
  reward_clip_model,
  reward_processor,
  reward_model,
  imgs,
  prompts,
  test_flag = False,
):
  if test_flag:
    image_pil = imgs
  else:
    image_pil = data_pipeline.numpy_to_pil(imgs)[0]

  device = next(reward_model.parameters()).device

  pixels = reward_processor(images = image_pil.convert("RGB"), return_tensors = "pt").pixel_values.to(weight_dtype).to(device)
  image_embeddings = reward_clip_model.get_image_features(pixels)

  inputs = reward_tokenizer(
    prompts,
    max_length = tokenizer.model_max_length,
    padding = "do_not_pad",
    truncation = True,
  )
  input_ids = inputs.input_ids
  padded_tokens = reward_tokenizer.pad(
    {"input_ids": input_ids},
    padding = True,
    return_tensors = "pt"
  )
  text_embeddings = reward_clip_model.get_text_features(
    input_ids = padded_tokens.input_ids.to(device).unsqueeze(0)
  )
  score = reward_model(text_embeddings, image_embeddings)
  return score.to(weight_dtype).squeeze(0).squeeze(0), text_embeddings.squeeze(0)



def calculate_reward_ir(
  model_pipeline,
  args,
  reward_tokenizer,
  tokenizer,
  weight_dtype,
  reward_clip_model,
  image_reward,
  imgs,
  prompts,
  test_flag = False,
):
  """Computes reward using ImageReward model."""
  if test_flag:
    image_pil = imgs
  else:
    image_pil = model_pipeline.numpy_to_pil(imgs)[0]
  blip_reward, _ = image_reward_get_reward(image_reward, image_pil, prompts, weight_dtype)
  if args.reward_filter == 1:
    blip_reward = torch.clamp(blip_reward, min = 0)

  inputs = reward_tokenizer(prompts, max_length = tokenizer.model_max_length, padding = "do_not_pad", truncation = True)
  input_ids = inputs.input_ids
  padded_tokens = reward_tokenizer.pad({"input_ids": input_ids}, padding = True, return_tensors = "pt")
  txt_emb = reward_clip_model.get_text_features(
    input_ids = padded_tokens.input_ids.to("cuda").unsqueeze(0)
  )
  return blip_reward.cpu().squeeze(0).squeeze(0), txt_emb.squeeze(0)



def get_batch(
  data_iter_loader,
  data_iterator,
  prompt_list,
  args,
  accelerator
):
  batch = next(data_iter_loader, None)
  if batch is None:
    data_iter_loader = iter(
      accelerator.prepare(
        data_iterator(prompt_list, batch_size = args.g_batch_size)
      )
    )
    batch = next(data_iter_loader)
  if args.single_prompt_flag == 1:
    for i in range(len(batch)):
      batch[i] = args.single_prompt
  batch_lst = []
  for i in range(len(batch)):
    batch_lst.extend([batch[i] for _ in range(args.num_samples)])
  return batch_lst, data_iter_loader

def trim_buffer(buffer_size, state_dict):
  if state_dict['state'].shape[0] > buffer_size:
    state_dict["prompt"] = state_dict["prompt"][-buffer_size:]
    state_dict["state"] = state_dict["state"][-buffer_size:]
    state_dict["next_state"] = state_dict["next_state"][-buffer_size:]
    state_dict["timestep"] = state_dict["timestep"][-buffer_size:]
    state_dict["final_reward"] = state_dict["final_reward"][-buffer_size:]
    state_dict["unconditional_prompt_embeds"] = \
        state_dict["unconditional_prompt_embeds"][-buffer_size:]
    state_dict["guided_prompt_embeds"] = state_dict["guided_prompt_embeds"][-buffer_size:]
    state_dict["text_embed"] = state_dict["text_embed"][-buffer_size:]
    state_dict["log_prob"] = state_dict["log_prob"][-buffer_size:]



def save_model(args, count, is_ddp, accelerator, unet):
  save_path = os.path.join(args.output_dir, f"save_{count}")

  print(f"saving model to {save_path}")
  if is_ddp:
    model = copy.deepcopy(accelerator.unwrap_model(unet)).to(torch.float32)
    model.save_attn_procs(save_path)
  else:
    model = copy.deepcopy(unet).to(torch.float32)
    model.save_attn_procs(save_path)
