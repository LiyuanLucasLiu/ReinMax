# The code has been modified from https://github.com/chijames/GST

import torch
import torch.nn.functional as F
from torch import nn
from model.categorical import categorical_repara

from functools import partial
import math

activation_map = {
    'leakyrelu': partial(nn.LeakyReLU, negative_slope=0.3),
}

class Encoder(nn.Module):
    def __init__(self, latent_dim, categorical_dim, activation='relu', input_size=784):
        super(Encoder, self).__init__()
        assert categorical_dim == 2
        self.latent_dim = latent_dim
        
        self.fc1 = nn.Linear(input_size, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, latent_dim)
        self.activation = activation_map[activation.lower()]()
        
    def forward(self, x):
        h1 = self.activation(self.fc1(x))
        h2 = self.activation(self.fc2(h1))
        p1 = self.fc3(h2).view(-1, self.latent_dim, 1)
        return torch.cat([p1, 1-p1], dim=-1)

class Decoder(nn.Module):
    def __init__(self, latent_dim, categorical_dim, activation='relu', input_size=784):
        super(Decoder, self).__init__()
        assert categorical_dim == 2
        self.fc1 = nn.Linear(latent_dim, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, input_size)
        self.activation = activation_map[activation.lower()]()
        self.sigmoid = nn.Sigmoid()
    
    def decode(self, logits):
        h1 = self.activation(self.fc1(logits))
        h2 = self.activation(self.fc2(h1))
        return self.sigmoid(self.fc3(h2)) 
        
    def forward(self, logits):
        return self.decode(logits)

class VAE(nn.Module):
    def __init__(
        self, 
        latent_dim=4, 
        categorical_dim=2, 
        temperature=1., 
        hard=True, 
        method='reinmax', 
        activation='relu',
        input_size=784
    ):
        super(VAE, self).__init__()
        assert categorical_dim == 2
        
        self.encoder = Encoder(latent_dim, categorical_dim, activation=activation, input_size=input_size)
        self.decoder = Decoder(latent_dim, categorical_dim, activation=activation, input_size=input_size)
        self.prior = nn.Parameter(torch.zeros(latent_dim))
        
        self.categorical_dim = categorical_dim
        self.latent_dim = latent_dim
        self.temperature = temperature
        self.hard = hard 
        self.method = method
        
        self.itensor = None

    def compute_code(self, data, with_log_p=False):
        theta = self.encoder(data)
        z, qy = categorical_repara(theta, self.temperature, self.method)
        qy = qy.view(data.size(0), self.latent_dim, self.categorical_dim)
        z = z.view(data.size(0), self.latent_dim, self.categorical_dim)
        z_first = z[:, :, 0]
        if with_log_p:
            log_y = (z * theta).sum(dim=-1) - torch.logsumexp(theta, dim=-1)
            return z_first, qy, log_y
        else:
            return z_first, qy
    
    def compute_bce_loss(self, data):
        z, _ = self.compute_code(data)
        loss = torch.nn.functional.binary_cross_entropy(
            self.decoder(z),
            data,
            reduction='none',
        ).sum() / data.size(0)
        return loss
    
    def forward(self, data):
        batch_size = data.size(0)
        z, qy = self.compute_code(data)
        BCE = torch.nn.functional.binary_cross_entropy(
            self.decoder(z),
            data,
            reduction='none',
        ).sum() / batch_size
        
        prior_dist = torch.sigmoid(self.prior)
        prior_dist = torch.cat((prior_dist, 1-prior_dist), dim=0).view(1, -1)
        log_prior_dist = torch.log(prior_dist + 1e-10)
        
        qy = qy.view(batch_size, -1)
        log_ratio = torch.log(qy + 1e-10)
        KLD = torch.sum(qy * (log_ratio- log_prior_dist), dim=-1).mean()
        return BCE, KLD, z, qy
    
    def approx_bce_gradient(self, data):
        self.train()
        
        if self.method == 'reinforce':
            z, _, log_y = self.compute_code(data, with_log_p=True)
            BCE = torch.nn.functional.binary_cross_entropy(
                self.decoder(z),
                data,
                reduction='none',
            ).sum() / data.size(0)
            log_p = log_y.sum(-1)
            loss = torch.sum(log_p * loss.detach()) / data.size(0)
        else:
            loss = self.compute_bce_loss(data).sum() / data.size(0)
        
        self.zero_grad()
        loss.backward()
        return self.theta_gradient
