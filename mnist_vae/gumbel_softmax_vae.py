# The code has been modified from https://github.com/chijames/GST.
# The code has been modified from pytorch example vae code and inspired by the origianl \
# tensorflow implementation of gumble-softmax by Eric Jang.

import argparse
import os
import sys
import numpy as np
import math
import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from model.vae import VAE

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--epochs', type=int, default=160, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--max-updates', type=int, default=0, metavar='N',
                        help='number of updates to train')
    parser.add_argument('--temperature', type=float, default=1.0, metavar='S',
                        help='softmax temperature')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--method', default='reinmax',
                        help='gumbel, st, rao_gumbel, gst-1.0, reinmax')
    parser.add_argument('--log-images', type=lambda x: str(x).lower()=='true', default=False, 
                        help='log the sample & reconstructed images')
    parser.add_argument('--lr', type=float, default=1e-3, 
                        help="learning rate for the optimizer")
    parser.add_argument('--latent-dim', type=int, default=128,
                        help="latent dimension")
    parser.add_argument('--categorical-dim', type=int, default=10,
                        help="categorical dimension")
    parser.add_argument('--optim', type=str, default='adam',
                        help="adam, radam")
    parser.add_argument('--activation', type=str, default='relu',
                        help="relu, leakyrelu")
    parser.add_argument('-s', '--gradient-estimate-sample', type=int, default=0,
                        help="number of samples used to estimate gradient bias (default 0: not estimate)")

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    print(args)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/MNIST', train=True, download=True,
                    transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/MNIST', train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    model = VAE(
        latent_dim = args.latent_dim, 
        categorical_dim = args.categorical_dim, 
        temperature = args.temperature, 
        method = args.method, 
        activation=args.activation
    )
    
    if args.cuda:
        model.cuda()
        
    if args.optim.lower() == 'adam':
        optimizer_class = optim.Adam
    else:
        assert args.optim.lower() == 'radam'
        optimizer_class = optim.RAdam
        
    optimizer = optimizer_class(model.parameters(), lr=args.lr)

    total_updates = 0
    
    try:
        for epoch in range(1, args.epochs + 1):
            
            if args.max_updates > 0 and total_updates >= args.max_updates:
                break
                
            ### Training ############
            model.train()
            train_loss, train_bce, train_kld = 0, 0, 0
            temp = args.temperature
            for batch_idx, (data, _) in enumerate(train_loader):
                data = data.view(data.size(0), -1)
                if args.cuda:
                    data = data.cuda()
                    
                if 0 == batch_idx and args.gradient_estimate_sample > 0:
                    _, qy = model.compute_code(data[:args.gradient_estimate_sample, :])
                    print('Entropy: {}'.format(torch.sum(qy * torch.log(qy + 1e-10), dim=-1).mean().item()))
                    
                    print('Method: {}'.format(model.method))
                    assert args.gradient_estimate_sample <= args.batch_size
                    rb0, rb1, bstd, cos = model.analyze_gradient(data[:args.gradient_estimate_sample, :], 1024)
                    print('Train Epoch: {} -- Training Epoch Relative Bias Ratio (w.r.t. exact gradient): {} '.format(epoch, rb0.item()))
                    print('Train Epoch: {} -- Training Epoch Relative Bias Ratio (w.r.t. approx gradient): {} '.format(epoch, rb1.item()))
                    print('Train Epoch: {} -- Training Epoch Relative Std (w.r.t. approx gradient): {} '.format(epoch, bstd.item()))
                    print('Train Epoch: {} -- Training Epoch COS SIM: {} '.format(epoch, cos.item()))
                    model.zero_grad()
                    
                    model.method = args.method
                
                bce, kld, _, qy = model(data)
                loss = bce + kld
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_updates += 1
                
                train_loss += loss.item() * len(data)
                train_bce += bce.item() * len(data)
                train_kld += kld.item() * len(data)
                if batch_idx % args.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)] \t Loss: {:.2f} \t BCE: {:.2f} \t KLD: {:.2f} \t Max of Softmax: {:.2f} +/- {:.2f} in [{:.2f} -- {:.2f}]'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                            100. * batch_idx / len(train_loader),
                            loss.item(),
                            bce.item(),
                            kld.item(),
                            qy.view(-1, args.categorical_dim).max(dim=-1)[0].mean(),
                            qy.view(-1, args.categorical_dim).max(dim=-1)[0].std(),
                            qy.view(-1, args.categorical_dim).max(dim=-1)[0].max(),
                            qy.view(-1, args.categorical_dim).max(dim=-1)[0].min(),
                        )
                    )
                    
            print('====> Epoch: {} Average loss: {:.6f} \t BCE: {:.6f} KLD: {:.6f}'.format(
                epoch, 
                train_loss / len(train_loader.dataset),
                train_bce / len(train_loader.dataset), 
                train_kld / len(train_loader.dataset),
            ))
            
            ### Testing ############
            model.eval()
            test_loss, test_bce, test_kld = 0, 0, 0
            temp = args.temperature
            for i, (data, _) in enumerate(test_loader):
                data = data.view(data.size(0), -1)
                if args.cuda:
                    data = data.cuda()
                bce, kld, (_, recon_batch), __ = model(data)
                test_loss += (bce + kld).item() * len(data)
                test_bce += bce.item() * len(data)
                test_kld += kld.item() * len(data)
                if i == 0 and args.log_images:
                    n = min(data.size(0), 8)
                    comparison = torch.cat([data[:n],
                                            recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                    save_image(comparison.data.cpu(),
                            'data/reconstruction_' + str(epoch) + '.png', nrow=n)

            print('====> Test set loss: {:.6f} \t BCE: {:.6f} \t KLD: {:.6f}'.format(
                test_loss / len(test_loader.dataset), 
                test_bce / len(test_loader.dataset),
                test_kld / len(test_loader.dataset)
            ))
            
            
            ### MISC ############
            
            if args.log_images:
                M = 64 * args.latent_dim
                np_y = np.zeros((M, args.categorical_dim), dtype=np.float32)
                np_y[range(M), np.random.choice(args.categorical_dim, M)] = 1
                np_y = np.reshape(np_y, [M // args.latent_dim, args.latent_dim, args.categorical_dim])
                sample = torch.from_numpy(np_y).view(M // args.latent_dim, args.latent_dim * args.categorical_dim)
                if args.cuda: sample = sample.cuda()
                sample = model.decoder.decode(sample).cpu()
                save_image(sample.data.view(M // args.latent_dim, 1, 28, 28),
                        'data/sample_' + str(epoch) + '.png')

    except Exception as e: 
        print(e)
        print("Error in Training, Failed")
        