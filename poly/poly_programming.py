import argparse

import torch
import torch.nn.functional as F
from torch import nn, optim
from model.categorical import categorical_repara

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training')
parser.add_argument('--train-step-per-epoch', type=int, default=200, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--epochs', type=int, default=40, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--temperature', type=float, default=1.0, metavar='S',
                    help='softmax temperature')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed')
parser.add_argument('--method', default='gumbel',
                    help='gumbel, st, rao_gumbel, gst-1.0, reinmax')
parser.add_argument('--lr', type=float, default=1e-3, 
                    help="learning rate for the optimizer")
parser.add_argument('--latent-dim', type=int, default=128,
                    help="latent dimension")
parser.add_argument('--pnorm', type=float, default=2,
                    help="p-norm would be used in the loss function")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

print(args)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

class Quadratic_Toy(nn.Module):
    def __init__(self):
        super(Quadratic_Toy, self).__init__()        
        self.theta = nn.Parameter(torch.Tensor(latent_dim, categorical_dim))
        self.theta.data.uniform_(-0.01, 0.01)


    def forward(self, temp, batch_size):
        theta_b = self.theta.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        if 'reinforce' == self.method:
            qy = F.softmax(theta_b, dim=-1)
            z = torch.distributions.one_hot_categorical.OneHotCategorical(logits=theta_b).sample()
            log_y = (z * theta_b).sum(dim=-1) - torch.logsumexp(theta_b, dim=-1)
        else: 
            z, qy = categorical_repara(theta_b, temp, self.method)
            log_y = None
        z = z.view(batch_size, -1, categorical_dim)[:, :, 0]
        qy = qy.view(batch_size, -1, categorical_dim)[:, :, 0]
        return z, qy, log_y


latent_dim = args.latent_dim
categorical_dim = 2  # one-of-K vector
targets = torch.Tensor([0.45]).repeat(args.latent_dim).contiguous()
batched_targets = targets.unsqueeze(0).expand(args.batch_size, -1)

model = Quadratic_Toy()
if args.cuda:
    model.cuda()
    targets = targets.cuda()
    batched_targets = batched_targets.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
model.method = args.method

def loss_scale(mse):
    return mse.abs().pow(args.pnorm)
    
# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(z):
    # MSE = (z - batched_targets).square().sum() / z.size(0) / latent_dim
    MSE = loss_scale(z - batched_targets).sum(dim = -1) / latent_dim
    # MSE = MSE.pow(1/4).sum() / z.size(0)
    MSE = MSE.sum() / z.size(0)

    return MSE

def reinforce_train(epoch):
    model.train()
    train_loss = 0
    temp = args.temperature
    for batch_idx in enumerate(range(args.train_step_per_epoch)):
        z, qy, log_y = model(temp, args.batch_size)
        # MSE = (z - batched_targets).square().sum(dim=-1) / latent_dim
        MSE = loss_scale(z - batched_targets).sum(dim=-1) / latent_dim
        log_p = log_y.sum(-1)
        loss = torch.sum(log_p * MSE.detach()) / z.size(0)

        optimizer.zero_grad()
        loss.backward()
        
        train_loss += (MSE.sum() / z.size(0)).item()
        optimizer.step()

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / args.train_step_per_epoch))

def exact_reinforce_train(epoch):
    model.train()
    train_loss = 0
    temp = args.temperature
    for batch_idx in enumerate(range(args.train_step_per_epoch)):
        qy = F.softmax(model.theta, dim=-1)
        # MSE_1 = (1 - batched_targets).square()
        # MSE_0 = batched_targets.square()
        MSE_1 = loss_scale(1 - batched_targets)
        MSE_0 = loss_scale(batched_targets)
        loss = MSE_1 * qy[:, 0] + MSE_0 * qy[:, 1]
        loss = loss.sum() / latent_dim
        
        optimizer.zero_grad()
        loss.backward()
        
        train_loss += (loss.sum()).item()
        optimizer.step()

def train(epoch):
    model.train()
    train_loss = 0
    temp = args.temperature
    for batch_idx in enumerate(range(args.train_step_per_epoch)):
        z, qy, _ = model(temp, args.batch_size)
        loss = loss_function(z)
        optimizer.zero_grad()
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / args.train_step_per_epoch))


def soft_train(epoch):
    model.train()
    train_loss = 0
    temp = args.temperature
    for batch_idx in enumerate(range(args.train_step_per_epoch)):
        z, qy, _ = model(temp, args.batch_size)
        loss = loss_function(z)
        optimizer.zero_grad()
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    qy = F.softmax(model.theta, dim=-1)
    # MSE_1 = (1 - targets).square()
    # MSE_0 = targets.square()
    MSE_1 = loss_scale(1 - targets)
    MSE_0 = loss_scale(targets)
    loss = MSE_1 * qy[:, 0] + MSE_0 * qy[:, 1]
    loss = loss.sum() / latent_dim
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, loss))

def run():
    for epoch in range(1, args.epochs + 1):
        if model.method == 'reinforce_exact':
            exact_reinforce_train(epoch)
        elif model.method == 'reinforce':
            reinforce_train(epoch)
        else:
            train(epoch)

if __name__ == '__main__':
    run()
