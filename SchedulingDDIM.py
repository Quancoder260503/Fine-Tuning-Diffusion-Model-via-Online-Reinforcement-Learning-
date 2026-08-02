from typing import Optional, Tuple, Union 
from diffusers import DDIMScheduler 
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput 
from diffusers.utils.torch_utils import randn_tensor 
import torch 
from torch.distributions import Normal 

class ExtendedDDIMScheduler(DDIMScheduler): 
  def get_variance_log_prob(self, timestep, prev_timestep): 
    alpha_prod_t = self.alphas_cumprod.to(timestep.device)[timestep] 
    mask_a = (prev_timestep >= 0).int().to(timestep.device) 
    mask_b = 1 - mask_a 
    alpha_prod_prev_t = self.alphas_cumprod.to(timestep.device)[prev_timestep] * mask_a +\
        self.final_alpha_cumprod.to(timestep.device) * mask_b
    
    # They try to make this a Markovian ? ye then it becomes DDPM in this way 
    beta_prod_t = 1 - alpha_prod_t 
    beta_prod_prev_t = 1 - alpha_prod_prev_t 
    variance = (beta_prod_prev_t / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_prev_t)
    return variance 

  def step_log_prob(
    self, 
    model_out, 
    timestep, 
    sample, 
    eta = 1., 
    use_clipped_model_output = False, 
    generator = None, 
    variance_noise = None,
    return_dict = True, 
  ): 
    """
    Args: 
      model_output (`torch.FloatTensor`) : direct output from learned diffusion model. 
      timestep (`int`) : current discrete timestep in the diffusion chain 
      sample (`torch.FloatTensor`) : current instance of sample being created by diffusion process 
      eta (`float`) : weight of the added noise in diffusion step 
      use_clipped_model_output (`bool`) : if `True`, compute "corrected" `model_output` from the 
      clipped predicted original sample.
      generator : rng 
      variance_noise (`torch.FloatTensor`) : instead of generating noise for the variance using rng, 
      we can directly provide the noise for the variance itself. 
      return_dict (`bool`) : option for returning tuple rather than DDIMSchedulerOutput class 
    Returns : 
      [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] or `tuple`:
      [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] if `return_dict` is \
        True, otherwise a `tuple`. When returning a tuple, the first element is the sample tensor.
    """

    if self.num_inference_steps is None : 
      raise ValueError("Number of inference steps is None, \
        run `set_timesteps` after creating the scheduler")
    
    prev_timestep = (timestep - self.config.num_train_timesteps // self.num_inference_steps)
    alpha_prod_t = self.alphas_cumprod.to(timestep.device)[timestep]
    mask_a = (prev_timestep >= 0).int().to(timestep.device) 
    mask_b = 1 - mask_a 
    
    # if the prev step over the limit then use the last step 
    alpha_prod_prev_t = self.alphas_cumprod.to(timestep.device)[prev_timestep] * mask_a +\
          self.final_alpha_cumprod.to(timestep.device) * mask_b
    
    beta_prod_t = 1 - alpha_prod_t 

    # predicted x_0

    if self.config.prediction_type == 'epsilon': 
      pred_original_sample = (sample - (beta_prod_t ** 0.5) * model_out) / (alpha_prod_t ** 0.5) 
    elif self.config.prediction_type == 'sample': 
      pred_original_sample = model_out
    elif self.config.prediction_type == 'v_prediction': 
      # get the next sample in the diffusion process ? 
      pred_original_sample =  (alpha_prod_t ** 0.5) * sample - (beta_prod_t ** 0.5) * model_out 
      model_out = (alpha_prod_t ** 0.5) * model_out + (beta_prod_t ** 0.5) * sample  
    else : 
      raise ValueError(f"prediction_type given as {self.config.prediction_type} must be one of \
       `sample`, `epsilon`, `v_prediction`")
    
    if self.config.clip_sample : 
      pred_original_sample = torch.clamp(pred_original_sample, -1, 1) 

    variance = self.get_variance_log_prob(timestep, prev_timestep).to(dtype = sample.dtype) 

    std_dev_t = (eta * variance ** (0.5)).to(dtype = sample.dtype) 

    if use_clipped_model_output : 
      model_out = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / (beta_prod_t ** 0.5) 

    pred_sample_direction = (1 - alpha_prod_prev_t - std_dev_t ** 2) ** 0.5 * model_out 

    prev_sample = alpha_prod_prev_t ** (0.5) * pred_original_sample + pred_sample_direction 
    
    log_prob = 0. 
    if eta > 0: 
      device = model_out.device
      if variance_noise is not None and generator is not None : 
        raise ValueError("Cannot pass both generator and variance_noise. Please make sure that \
          exactly one of them stays None")
      
      if variance_noise is None: 
        variance_noise = randn_tensor(
          model_out.shape, 
          generator = generator, 
          device = device, 
          dtype = model_out.dtype, 
        )
      variance = std_dev_t * variance_noise 
      dist = Normal(prev_sample, std_dev_t) 
      prev_sample = prev_sample.detach().clone() + variance 

      log_prob = dist.log_prob(prev_sample.detach().clone())\
        .mean(dim = -1).mean(dim = -1).mean(dim = -1).detach().cpu() 

    if not return_dict : 
      return (prev_sample, ) 
    

    return (DDIMSchedulerOutput(prev_sample = prev_sample, pred_original_sample = pred_original_sample), log_prob) 
  
  def step_forward_logprob(
    self, 
    model_out, 
    timestep, 
    sample, 
    next_sample, 
    eta = 1., 
    use_clipped_model_output = False, 
    generator = None, 
    variance_noise = None, 
    return_dict = True, 
  ):
    """
    Predict the sample at previous timestep by reversing the SDE 
    Args: 
      model_output (`torch.FloatTensor`) : direct output from learned diffusion model. 
      timestep (`int`) : current discrete timestep in the diffusion chain 
      sample (`torch.FloatTensor`) : current instance of sample (x_t) being created by diffusion process 
      next_sample (`torch.FloatTensor`) : instance of the next sample (x_{t - 1}) being created by diffusion
      process. Reinforcement learning sampling in the backward process, thus, $x_{t - 1}$ is the "next" is 
      the very next sample of $x_t$. 

      eta (`float`) : weight of the added noise in diffusion step 
      use_clipped_model_output (`bool`) : if `True`, compute "corrected" `model_output` from the 
      clipped predicted original sample.
      generator : rng 
      variance_noise (`torch.FloatTensor`) : instead of generating noise for the variance using rng, 
      we can directly provide the noise for the variance itself. 
      return_dict (`bool`) : option for returning tuple rather than DDIMSchedulerOutput class 
    Returns : 
      log_probobality 
    """
    if self.num_inference_steps is None : 
      raise ValueError("Number of inference steps is None, \
        run `set_timesteps` after creating the scheduler")
    
    prev_timestep = (timestep - self.config.num_train_timesteps // self.num_inference_steps)

    alpha_prod_t = self.alphas_cumprod.to(timestep.device)[timestep]
    mask_a = (prev_timestep >= 0).int().to(timestep.device) 
    mask_b = 1 - mask_a 
    
    # if the prev step over the limit then use the last step 
    alpha_prod_prev_t = self.alphas_cumprod.to(timestep.device)[prev_timestep] * mask_a +\
          self.final_alpha_cumprod.to(timestep.device) * mask_b
    
    beta_prod_t = 1 - alpha_prod_t 

    # predicted x_0

    if self.config.prediction_type == 'epsilon': 
      pred_original_sample = (sample - (beta_prod_t ** 0.5) * model_out) / (alpha_prod_t ** 0.5) 
    elif self.config.prediction_type == 'sample': 
      pred_original_sample = model_out
    elif self.config.prediction_type == 'v_prediction': 
      # get the next sample in the diffusion process ? 
      pred_original_sample =  (alpha_prod_t ** 0.5) * sample - (beta_prod_t ** 0.5) * model_out 
      model_out = (alpha_prod_t ** 0.5) * model_out + (beta_prod_t ** 0.5) * sample  
    else : 
      raise ValueError(f"prediction_type given as {self.config.prediction_type} must be one of \
       `sample`, `epsilon`, `v_prediction`")
    
    if self.config.clip_sample : 
      pred_original_sample = torch.clamp(pred_original_sample, -1, 1) 

    variance = self.get_variance_log_prob(timestep, prev_timestep).to(dtype = sample.dtype) 

    std_dev_t = (eta * variance ** (0.5)).to(dtype = sample.dtype) 

    if use_clipped_model_output : 
      model_out = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / (beta_prod_t ** 0.5) 

    pred_sample_direction = (1 - alpha_prod_prev_t - std_dev_t ** 2) ** 0.5 * model_out 

    prev_sample = alpha_prod_prev_t ** (0.5) * pred_original_sample + pred_sample_direction 
    
    log_prob = 0. 
    if eta > 0: 
      device = model_out.device
      if variance_noise is not None and generator is not None : 
        raise ValueError("Cannot pass both generator and variance_noise. Please make sure that \
          exactly one of them stays None")
      
      if variance_noise is None: 
        variance_noise = randn_tensor(
          model_out.shape, 
          generator = generator, 
          device = device, 
          dtype = model_out.dtype, 
        )
      variance = std_dev_t * variance_noise 
      dist = Normal(prev_sample, std_dev_t) 
      prev_sample = prev_sample.detach().clone() + variance 

      log_prob = dist.log_prob(next_sample.detach().clone())\
        .mean(dim = -1).mean(dim = -1).mean(dim = -1).detach().cpu() 
    return log_prob 
    

    