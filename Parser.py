


def set_seed(seed):
  """Sets seed."""
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  os.environ['TOKENIZERS_PARALLELISM'] = 'false'
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  torch.multiprocessing.set_sharing_strategy('file_system')

def parse_args(): 
  parser = argparse.ArgumentParser(description = "Training arguments for Diffusion PPO") 
  parser.add_argument("--pretrained_model_path", type = str, default="runwayml/stable-diffusion-v1-5", 
    help = ("Path to pretrained model or model identifier from "" huggingface.co/models."),
  )
  parser.add_argument("--revision", type = str, default = None, required = False,
    help = ("Revision of pretrained model identifier from huggingface.co/models."),
  )
  parser.add_argument("--dataset_name", type = str, default = None)
  parser.add_argument("--dataset_config_name", type = str, default = None,
    help = ("The config of the Dataset, leave as None if there's only one config."),
  )
  parser.add_argument("--train_data_dir", type = str, default = None,
    help=(
    "A folder containing the training data. Folder contents must follow"
    " the structure described in"
    " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In"
    " particular, a `metadata.jsonl` file must exist to provide the"
    " captions for the images. Ignored if `dataset_name` is specified."
    ),
  )
  parser.add_argument("--image_column", type = str, default = "image",
    help = "The column of the dataset containing an image.",
  )
  parser.add_argument("--caption_column", type = str, default = "text",
    help = "The column of the dataset containing a caption or a list of captions."
  )
  parser.add_argument("--max_train_samples", type = int, default = None,
    help = "For debugging purposes or quicker training, truncate the number of \
      training examples to this value if set."
  )
  parser.add_argument("--output_dir", type = str, default = "online_model",
    help = "The output directory where the model predictions and checkpoints will be written."
  )
  parser.add_argument("--cache_dir", type = str, default = None, 
    help = "The directory where the downloaded models and datasets will be stored."
  )
  parser.add_argument("--seed", type = int, default = None, 
    help = "A seed for reproducible training."
  )
  parser.add_argument("--resolution", type = int, default = 512,
    help = "The resolution for input images, all the images in the \
      train/validation dataset will be resized to this resolution"
  )
  parser.add_argument("--center_crop", default = True, action="store_true",
    help = (
        "Whether to center crop the input images to the resolution. If not"
        " set, the images will be randomly cropped. The images will be"
        " resized to the resolution first before cropping."
    ),
  )
  parser.add_argument("--random_flip", default = True, action = "store_true", help = "whether to randomly flip images horizontally")
  
  parser.add_argument("--train_batch_size", type = int, default = 8,
    help = "Batch size (per device) for the training dataloader.",
  )
  parser.add_argument("--num_train_epochs", type = int, default = 100)
  parser.add_argument("--max_train_steps", type = int, default = 10000,
    help = "Total number of training steps to perform. If provided, overrides num_train_epochs."
  )
  parser.add_argument("--gradient_checkpointing", default = True, action = "store_true",
    help = "Whether or not to use gradient checkpointing to \
      save memory at the expense of slower backward pass."
  )

  parser.add_argument("--scale_lr", action = "store_true", default = False,
    help = "Scale the learning rate by the number of GPUs, \
      gradient accumulation steps, and batch size."
  )
  parser.add_argument("--lr_scheduler", type = str, default = "constant",
    help = 'The scheduler type to use. Choose between ["linear", "cosine", \
        "cosine_with_restarts", "polynomial", "constant","constant_with_warmup"]'
  )
  parser.add_argument("--lr_warmup_steps", type = int, default = 0,
    help = "Number of steps for the warmup in the lr scheduler.",
  )
  parser.add_argument("--use_8bit_adam", action = "store_true",
    help = "Whether or not to use 8-bit Adam from bitsandbytes.",
  )
  parser.add_argument("--allow_tf32", action = "store_true",
    help = "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training.\
      For more information, see https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
  )
  parser.add_argument("--use_ema", default = False, help = "Whether to use EMA model.")
  parser.add_argument("--non_ema_revision", type = str, default = None, required = False,
    help = "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier \
      of the local or remote repository specified with --pretrained_model_name_or_path."
  )
  parser.add_argument("--dataloader_num_workers", type = int, default = 0,
    help = "Number of subprocesses to use for data loading. 0 means that \
       the data will be loaded in the main process."
  )
  parser.add_argument("--adam_beta1", type = float, default = 0.9, help = "The beta1 parameter for the Adam optimizer.")
  parser.add_argument("--adam_beta2",type = float, default = 0.999, help = "The beta2 parameter for the Adam optimizer.")
  parser.add_argument("--adam_weight_decay", type = float, default = 0.0, help = "Weight decay to use.")
  parser.add_argument("--adam_epsilon", type = float, default = 1e-08, help = "Epsilon value for the Adam optimizer")
  parser.add_argument("--max_grad_norm", default = 1.0, type = float, help = "Max gradient norm.")
  parser.add_argument("--push_to_hub", action = "store_true", help = "Whether or not to push the model to the Hub.")

  parser.add_argument("--hub_token", type = str, default = None,
    help = "The token to use to push to the Model Hub.",
  )
  parser.add_argument("--hub_model_id", type = str, default = None,
    help = "The name of the repository to keep in sync with the local `output_dir`."
  )
  parser.add_argument("--logging_dir", type = str, default = "logs",
    help = "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. \
      Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
  )
  parser.add_argument("--mixed_precision", type = str, default = "fp16", choices = ["no", "fp16", "bf16"],
    help = "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10.and an Nvidia Ampere GPU. \
        Default to the value of accelerate config of the current system or the flag passed with the `accelerate.launch` command.\
        Use this argument to override the accelerate config." 
  )
  parser.add_argument("--report_to", type = str, default = "tensorboard",
    help = "The integration to report the results and logs to. Supported platforms\
      are `tensorboard` (default), `wandb` and `comet_ml`. Use `all` to report to all integrations."
  )
  parser.add_argument("--local_rank", type = int, default = -1, help = "For distributed training: local_rank")
  parser.add_argument("--checkpointing_steps", type = int, default = 2000,
    help = "Save a checkpoint of the training state every X updates. These checkpoints \
        are only suitable for resuming training using `--resume_from_checkpoint`."
  )
  parser.add_argument("--checkpoints_total_limit", type = int, default = None,
    help = "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`. \
       See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state for more docs"
  )
  parser.add_argument("--resume_from_checkpoint", type = str, default = None,
    help = "Whether training should be resumed from a previous checkpoint. Use a 'path saved by \
    `--checkpointing_steps`, or `latest` to automatically select the last available checkpoint"
  )
  parser.add_argument("--enable_xformers_memory_efficient_attention", action = "store_true",
    help = "Whether or not to use xformers.",
  )
  parser.add_argument("--policy_step", type = int, default = 5,
    help = "The number of steps to update the policy per sampling step",
  )
  parser.add_argument("--policy_batch_size", type = int, default = 2,
    help = "batch size for policy update per gpu, before gradient accumulation; \
      total batch size per gpu = gradient_accumulation_steps * policy_batch_size"
    
  )
  parser.add_argument("--value_flag", type = int, default = 1)
  parser.add_argument("--g_step", type = int, default = 1, help = "The number of sampling steps")
  parser.add_argument("--g_batch_size", type = int, default = 6,
    help = "batch size of prompts for sampling per gpu",
  )
  parser.add_argument("--sft_path", type = str, default = "./checkpoints/models/stf_model",
    help = "path to the pretrained supervised finetuned model",
  )
  parser.add_argument("--reward_model_path", type = str, default = "./checkpoints/reward/reward_model.pkl",
    help = "path to the pretrained reward model",
  )
  parser.add_argument("--reward_weight", type = float, default = 100, help = "weight of reward loss")
  parser.add_argument("--reward_flag", type = int, default = 0, help = "0: ImageReward, 1: Custom reward model")
  parser.add_argument("--reward_filter", type = int, default = 0, help = "0: raw value, 1: took positive")
  parser.add_argument("--kl_weight", type = float, default = 0.01, help = "weight of kl loss")
  parser.add_argument("--kl_warmup", type = int, default = -1, help = "warm up for kl weight")
  parser.add_argument("--buffer_size", type = int, default = 1000, help = "size of replay buffer")
  parser.add_argument("--value_batch_size", type = int, default = 256, 
    help = "batch size for value function update per gpu, no gradient accumulation"  
  )
  parser.add_argument("--value_lr", type = float, default = 1e-4, help = "learning rate for value fn")
  parser.add_argument("--value_step", type = int, default = 5,
    help = "The number of steps to update the value function per sampling step"
  )
  parser.add_argument("--value_gradient_step", type = int, default = 5,
    help = "Divisor applied to the value loss before backward (e.g. for accumulation across value_step)."
  )
  parser.add_argument("--save_interval", type = int, default = 100, help = "save model every save_interval steps")
  parser.add_argument("--num_samples", type = int, default = 1, help = "number of samples to generate per prompt")
  parser.add_argument("--clip_norm", type = float, default = 0.1, help = "norm for gradient clipping")
  parser.add_argument("--gradient_accumulation_steps", type = int, default = 12,
    help = "Number of updates steps to accumulate before performing\
     a backward/update pass for policy"
  )
  parser.add_argument("--lora_rank", type=int, default=4, help="rank for LoRA")
  parser.add_argument("--learning_rate", type = float, default = 1e-5, help = "Learning rate for policy")
  parser.add_argument("--prompt_path", type = str, default = "./dataset/drawbench/data_meta.json",
    help = "path to the prompt dataset",
  )
  parser.add_argument("--prompt_category", type = str, default = "all",
    help = "all or specific categories with comma [e.g., color,count]",
  )
  parser.add_argument("--single_prompt_flag", type = int, default = 1)
  parser.add_argument("--single_prompt", type = str, default = "A green colored rabbit.")
  parser.add_argument("--sft_initialization", type = int, default = 0)
  parser.add_argument("--num_validation_images", type = int, default = 2)
  parser.add_argument("--multi_gpu", type = int, default = 0)
  parser.add_argument("--ratio_clip", type = float, default = 1e-4)
  
  args = parser.parse_args()
  env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
  if env_local_rank != -1 and env_local_rank != args.local_rank:
    args.local_rank = env_local_rank

  # # Sanity checks
  # if args.dataset_name is None and args.train_data_dir is None:
  #     raise ValueError("Need either a dataset name or a training folder.")

  # default to using the same revision for the non-ema model if not specified
  if args.non_ema_revision is None:
    args.non_ema_revision = args.revision

  return args