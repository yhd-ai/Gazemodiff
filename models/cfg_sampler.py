import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

# A wrapper model for Classifier-free guidance **SAMPLING** only
# https://arxiv.org/abs/2207.12598
class ClassifierFreeSampleModel(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model  # model is the actual model to run

        self.num_frames = self.model.num_frames
        self.latent_dim = self.model.latent_dim
        self.num_layers = self.model.num_layers
        self.num_heads = self.model.num_heads
        self.ff_size = self.model.ff_size
        self.dropout = self.model.dropout
        self.activation = self.model.activation
        self.input_feats = self.model.input_feats
        self.time_embed_dim = self.model.latent_dim

    def forward(self, x, timesteps, y=None):
        cond_mode = self.model.cond_mode

        y_uncond = deepcopy(y)
        
        out = self.model(x, timesteps, y)
        out_uncond = self.model(x, timesteps, y_uncond)
        return out_uncond + (y['scale'].view(-1, 1, 1, 1) * (out - out_uncond))

