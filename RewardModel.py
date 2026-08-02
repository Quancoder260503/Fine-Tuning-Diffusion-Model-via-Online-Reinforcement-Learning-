import numpy as np 
import torch 
from torch import nn 
import torch.nn.functional as F 

def get_activation(act_name): 
  if act_name == 'gelu':
    return nn.GELU() 
  elif act_name == 'relu':
    return nn.ReLU() 
  elif act_name == 'leaky_relu': 
    return nn.LeakyReLU() 
  elif act_name == 'tanh' : 
    return nn.Tanh() 
  elif act_name == 'sig': 
    return nn.Sigmoid() 
  else : 
    raise NotImplementedError(f"activation {act_name} not supported !")
            

def network(in_size, out_size, h_size, num_layers, activation): 
  network = [] 
  for _ in range(num_layers): 
    network.append(nn.Linear(in_size, h_size))
    network.append(nn.GELU(approximate = 'tanh'))  
    in_size = h_size 
  network.append(nn.Linear(in_size, out_size))
  network.append(get_activation(activation))
  return network 

class RewardModel(nn.Module): 
  def __init__(
   self, 
   in_size = 1024, 
   h_size = 512, 
   num_layers = 3, 
   activation = 'relu'
  ):
    super(RewardModel, self).__init__() 
    self.in_size = in_size 
    self.h_size = h_size 
    self.num_layers = num_layers 
    self.activation = activation 

    self.model = nn.Sequential(
      *network( 
        in_size = in_size, 
        out_size = 1, 
        h_size = h_size, 
        num_layers = num_layers, 
        activation = activation, 
      )
    )
  
  def forward(self, text, img): 
    input = torch.cat([text, img], axis = -1) 
    score = self.model(input)
    return score 
  
class RewardModelAdapter(nn.Module): 
  def __init__(
   self, 
   vision_in_size, 
   text_in_size, 
   out_size, 
   h_size = 512, 
   num_layers = 3, 
   activation = 'relu', 
  ):
    super(RewardModelAdapter, self).__init__() 
    self.vision_model = nn.Sequential( 
      *network(
        in_size = vision_in_size, 
        h_size = h_size, 
        out_size = out_size, 
        num_layers = num_layers, 
        activation = activation, 
      )
    )
    self.text_model = nn.Sequential(
      *network(
        in_size = text_in_size, 
        h_size = h_size, 
        out_size = out_size, 
        num_layers = num_layers, 
        activation = activation, 
      )
    )

  def forward(self, text, img): 
    vis_out = self.vision_model(img)
    text_out = self.text_model(text) 
    return vis_out, text_out
   

class ConditionalLinear(nn.Module): 
  def __init__(self, input_dim, output_dim, num_diffusion_steps):
    super(ConditionalLinear, self).__init__() 
    self.input_dim = input_dim 
    self.output_dim = output_dim 
    self.num_diffusion_steps = num_diffusion_steps 
    
    self.linear_1 = nn.Linear(input_dim, output_dim) 
    self.embed_1 = nn.Embedding(num_diffusion_steps, output_dim) 
    self.embed_1.weight.data.uniform_()
    torch.nn.init.xavier_normal_(self.linear_1.weight) 

  def forward(self, x, y): 
    out = self.linear_1(x) 
    gamma = self.embed_1(y) 
    out = out * gamma.view(-1, self.output_dim)
    return out 
  

class ValueApproximator(nn.Module): 
  def __init__(self, num_diffusion_steps, img_shape, layer_dims, activation):
    super(ValueApproximator, self).__init__() 
    self.num_diffusion_steps = num_diffusion_steps 
    self.img_shape = img_shape 
    self.num_layers = len(layer_dims)
    self.linear_0 = ConditionalLinear(int(np.prod(img_shape)), layer_dims[0], num_diffusion_steps) 
    for i in range(1, len(layer_dims)): 
      setattr(self, f"linear_{i}", ConditionalLinear(
        input_dim = layer_dims[i - 1], 
        output_dim = layer_dims[i], 
        num_diffusion_steps = num_diffusion_steps
      ))
    self.final_layer = nn.Linear(layer_dims[-1], 1) 
    torch.nn.init.xavier_normal_(self.final_layer.weight) 

    self.act = get_activation(activation)

  def forward(self, img, time): 
    x = img.view(img.shape[0], -1) 
    x = self.act(self.linear_0(x, time))
    for i in range(1, self.num_layers):  
      x = getattr(self, f"linear_{i}")(x, time) 
      x = self.act(x) 
    return self.final_layer(x)

class CombineValueApproximator(nn.Module): 
  def __init__(self, num_diffusion_steps, img_shape, text_dim, layer_dims, activation):
    super(CombineValueApproximator, self).__init__() 
    self.num_diffusion_steps = num_diffusion_steps 
    self.img_shape = img_shape 
    self.num_layers = len(layer_dims)
    self.linear_0 = ConditionalLinear(int(np.prod(img_shape)) + text_dim, layer_dims[0], num_diffusion_steps) 
    for i in range(1, len(layer_dims)): 
      setattr(self, f"linear_{i}", ConditionalLinear(
        input_dim = layer_dims[i - 1], 
        output_dim = layer_dims[i], 
        num_diffusion_steps = num_diffusion_steps
      ))
    self.final_layer = nn.Linear(layer_dims[-1], 1) 
    torch.nn.init.xavier_normal_(self.final_layer.weight) 
    self.act = get_activation(activation)

  def forward(self, img, text_emb, time): 
    x = img.view(img.shape[0], -1) 
    x = torch.cat([x, text_emb], dim = -1)
    x = self.act(self.linear_0(x, time))
    for i in range(1, self.num_layers):  
      x = getattr(self, f"linear_{i}")(x, time) 
      x = self.act(x) 
    return self.final_layer(x)
  


class TimeEmbedding(nn.Module): 
  def __init__(self, num_diffusion_steps, embed_dim):
    super(TimeEmbedding, self).__init__() 
    self.embed_dim = embed_dim
    self.num_diffusion_steps = num_diffusion_steps 
    self.embedding = nn.Embedding(num_diffusion_steps, embed_dim)

  def forward(self, time): 
    time_embedding = self.embedding(time)
    time_embedding = time_embedding.view(-1, self.embed_dim)
    return time_embedding 

class TDConvNet(nn.Module): 
  def __init__(self, time_dim, max_time, embed_dim, num_classes):
    super(TDConvNet, self).__init__() 
    self.time_dim = time_dim 
    self.max_time = max_time
    self.embed_dim = embed_dim 
    self.num_classes = num_classes
    # might replace the learnable embedding by sinusoidal 
    self.time_embedding = TimeEmbedding(max_time, embed_dim)

    self.conv_1 = nn.Conv1d(in_channels = 1, out_channels = 16, kernel_size = 3) 
    self.conv_2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3) 
    self.fc_1 = nn.Linear(32 * ((time_dim - 4) // 2) ** 2 + embed_dim, 64)
    self.fc_2 = nn.Linear(64, num_classes)

  def forward(self, x, time): 
    """
    Args : 
      x (`torch.Tensor`) of shape [batch_size, 1, time_dim] 
      time (`torch.Tensor`) of shape [batch_size, 1] 
    """  
    x = torch.relu(self.conv_1(x))
    x = torch.max_pool1d(x, 2) 
    x = torch.relu(self.conv_2(x)) 
    x = torch.max_pool1d(x, 2) 
    x = x.view(-1, 32 * ((self.time_dim - 4) // 2) ** 2) 
    time_embed = self.time_embedding(time) 
    x = torch.cat([x, time_embed], dim = 1) 
    x = torch.relu(self.fc_1(x)) 
    x = self.fc_2(x) 
    return x 
