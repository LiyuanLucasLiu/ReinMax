# The code has been modified from https://github.com/chijames/GST

import torch
import torch.nn.functional as F
from torch import nn
from model.categorical import categorical_repara

import math

activation_map = {
    'relu': nn.ReLU,
    'leakyrelu': nn.LeakyReLU,
}

class Encoder(nn.Module):
    def __init__(self, latent_dim, categorical_dim, activation='relu'):
        super(Encoder, self).__init__()
        self.categorical_dim = categorical_dim
        self.latent_dim = latent_dim
        
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, latent_dim * categorical_dim)
        self.activation = activation_map[activation.lower()]()
        
    def forward(self, x):
        h1 = self.activation(self.fc1(x))
        h2 = self.activation(self.fc2(h1))
        # Note that no activation function is applied to the output of encoder 
        # this is consistent with the original categorical MNIST VAE as in
        # https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
        # Intuitively, applying activation function like ReLU on encoder output is not
        # recommended, since:
        # 1. softmax function is a non-linear transformation itself;
        # 2. as the output of softmax always sums up to one, the gradient on its input 
        #    would sum up to zero. applying additional activation functions like ReLU woul
        #    break this structure, leading to a sub-optimal performance. 
        return self.fc3(h2).view(-1, self.latent_dim, self.categorical_dim)

class Decoder(nn.Module):
    def __init__(self, latent_dim, categorical_dim, activation='relu'):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim * categorical_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 784)
        self.activation = activation_map[activation.lower()]()
        self.sigmoid = nn.Sigmoid()
    
    def decode(self, logits):
        h1 = self.activation(self.fc1(logits))
        h2 = self.activation(self.fc2(h1))
        return self.sigmoid(self.fc3(h2))
        
    def forward(self, logits, target):
        return torch.nn.functional.binary_cross_entropy(
            self.decode(logits),
            target,
            reduction='none',
        )

class VAE(nn.Module):
    def __init__(
        self, 
        latent_dim=4, 
        categorical_dim=2, 
        temperature=1., 
        method='reinmax', 
        activation='relu'
    ):
        super(VAE, self).__init__()
        
        self.encoder = Encoder(latent_dim, categorical_dim, activation=activation)
        self.decoder = Decoder(latent_dim, categorical_dim, activation=activation)
        
        self.categorical_dim = categorical_dim
        self.latent_dim = latent_dim
        self.temperature = temperature
        self.method = method
        
        if 'exact' == self.method:
            self.forward = self.forward_exact
        else:
            self.forward = self.forward_approx
        self.itensor = None
        self.compute_code = self.compute_code_regular

    def compute_code_regular(self, data, with_log_p=False):
        theta = self.encoder(data)
        z, qy = categorical_repara(theta, self.temperature, self.method)
        qy = qy.view(data.size(0), self.latent_dim, self.categorical_dim)
        z = z.view(data.size(0), self.latent_dim, self.categorical_dim)
        if with_log_p:
            log_y = (z * theta).sum(dim=-1) - torch.logsumexp(theta, dim=-1)
            return z, qy, log_y
        else:
            return z, qy
        
    def compute_code_track(self, data, with_log_p=False):
        theta = self.encoder(data)
        def theta_gradient_save(gradient):
            self.theta_gradient = gradient 
            return gradient 
        theta.register_hook(theta_gradient_save)
        z, qy = categorical_repara(theta, self.temperature, self.method)
        qy = qy.view(data.size(0), self.latent_dim, self.categorical_dim)
        z = z.view(data.size(0), self.latent_dim, self.categorical_dim)
        if with_log_p:
            log_y = (z * theta).sum(dim=-1) - torch.logsumexp(theta, dim=-1)
            return z, qy, log_y
        else:
            return z, qy
    
    def compute_bce_loss(self, data):
        z, _ = self.compute_code(data)
        loss = self.decoder(z.view(data.size(0), -1), data)
        return loss
    
    def forward_approx(self, data):
        batch_size = data.size(0)
        z, qy = self.compute_code(data)
        r_d = z.view(batch_size, -1)
        BCE = self.decoder(r_d, data).sum() / batch_size
        
        qy = qy.view(batch_size, -1)
        log_ratio = torch.log(qy + 1e-10)
        KLD = torch.sum(qy * log_ratio, dim=-1).mean() + math.log(self.categorical_dim) * self.latent_dim
        return BCE, KLD, (z, r_d), qy
    
    def exact_bce_loss(self, data):
        def convert_to_i_base(i):
            i_list = list()
            while i > 0:
                i_list.append(i % self.categorical_dim)
                i = i // self.categorical_dim
            i_list = [0] * (self.latent_dim - len(i_list)) + i_list[::-1]
            return i_list
        
        search_size = self.categorical_dim ** self.latent_dim
        assert search_size < 16384, "categorical_dim ** latent_dim too large"
        with torch.no_grad():
            if self.itensor is None:
                i_tensor_list = list()
                for i in range(search_size):
                    i_list = convert_to_i_base(i)
                    i_tensor_list.append(
                        torch.Tensor(
                            [
                                [i_i == j for j in range(self.categorical_dim)]
                                for i_i in i_list
                            ]
                        ).view(-1)
                    )
                self.itensor = torch.stack(i_tensor_list).view(search_size, -1).to(data.device)
                
            i_tensor = self.itensor.unsqueeze(0).expand(data.size(0), -1, -1) # batchsize, sample, z_logits
            target = data.unsqueeze(1).expand(-1, search_size, -1) # batchsize, sample, i_logits
            i_loss = self.decoder(i_tensor, target).detach()
        
        _, qy = self.compute_code(data)
        qy = qy.view(-1, self.latent_dim * self.categorical_dim).unsqueeze(1).expand(-1, search_size, -1)
        qy = (i_tensor * qy).view(-1, search_size, self.latent_dim, self.categorical_dim).sum(dim=-1)
        loss = (torch.prod(qy, dim=-1) * i_loss.sum(dim=-1)).sum() / data.size(0)
        
        return loss 

    def forward_exact(self, data):
        def convert_to_i_base(i):
            i_list = list()
            while i > 0:
                i_list.append(i % self.categorical_dim)
                i = i // self.categorical_dim
            i_list = [0] * (self.latent_dim - len(i_list)) + i_list[::-1]
            return i_list
        
        batch_size = data.size(0)
        search_size = self.categorical_dim ** self.latent_dim
        with torch.no_grad():
            if self.itensor is None:
                i_tensor_list = list()
                for i in range(search_size):
                    i_list = convert_to_i_base(i)
                    i_tensor_list.append(
                        torch.Tensor(
                            [
                                [i_i == j for j in range(self.categorical_dim)]
                                for i_i in i_list
                            ]
                        ).view(-1)
                    )
                self.itensor = torch.stack(i_tensor_list).view(search_size, -1).to(data.device)
                
            i_tensor = self.itensor.unsqueeze(0).expand(batch_size, -1, -1) # batchsize, sample, z_logits
            target = data.unsqueeze(1).expand(-1, search_size, -1) # batchsize, sample, i_logits
            i_loss = self.decoder(i_tensor, target).detach()
        
        z, qy = self.compute_code(data)
        i_qy = qy.view(-1, self.latent_dim * self.categorical_dim).unsqueeze(1).expand(-1, search_size, -1)
        i_qy = (i_tensor * i_qy).view(-1, search_size, self.latent_dim, self.categorical_dim).sum(dim=-1)
        BCE_ENC = (torch.prod(i_qy, dim=-1) * i_loss.sum(dim=-1)).sum() / batch_size
        
        qy = qy.view(batch_size, -1)
        log_ratio = torch.log(qy + 1e-10)
        KLD = torch.sum(qy * log_ratio, dim=-1).mean() + math.log(self.categorical_dim) * self.latent_dim
        
        r_d = z.detach().view(batch_size, -1)
        BCE_DEC = self.decoder(r_d, data).sum() / batch_size
        return BCE_ENC - BCE_ENC.detach() + BCE_DEC, KLD, (z, r_d), qy
    
    def exact_bce_gradient(self, data):
        self.train()
        
        loss = self.exact_bce_loss(data)
        
        self.zero_grad()
        loss.backward()
        return self.theta_gradient

    def approx_bce_gradient(self, data):
        self.train()
        
        if self.method == 'reinforce':
            z, _, log_y = self.compute_code(data, with_log_p=True)
            loss = self.decoder(z.view(data.size(0), -1), data)
            log_p = log_y.sum(-1)
            loss = torch.sum(log_p * loss.detach()) / data.size(0)
        else:
            loss = self.compute_bce_loss(data).sum() / data.size(0)
        
        self.zero_grad()
        loss.backward()
        return self.theta_gradient

    def analyze_gradient(self, data, ct):
        self.compute_code = self.compute_code_track
        exact_grad = self.exact_bce_gradient(data)
        
        mean_grad = torch.zeros_like(exact_grad).double()
        std_grad = torch.zeros_like(exact_grad).double()
        
        for i in range(ct):
            grad = self.approx_bce_gradient(data)
        
            mean_grad += grad 
            std_grad += grad ** 2

        self.compute_code = self.compute_code_regular
        mean_grad = mean_grad / ct 
        std_grad = (std_grad / ct - mean_grad ** 2).abs() ** 0.5 
        

        diff = (exact_grad - mean_grad).norm()
        return (
            diff / exact_grad.norm(), 
            diff / mean_grad.norm(), 
            std_grad.norm() / mean_grad.norm(), 
            (exact_grad * mean_grad).sum() / (exact_grad.norm() * mean_grad.norm())
        )
