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
from model.vae_rodeo import VAE

def dynamic_binalization(x, y):
    return (torch.rand(x.size()) < x).type_as(x), y

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--max-updates', type=int, default=1000000, metavar='N',
                        help='number of updates to train (default: 10)')
    parser.add_argument('--temperature', type=float, default=1.0, metavar='S',
                        help='softmax temperature (default: 1.0)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--hard', type=str, default='True',
                        help='hard Gumbel Softmax')
    parser.add_argument('--method', default='reinmax',
                        help='gumbel, st, rao_gumbel, gst-1.0, reinmax')
    parser.add_argument('--log-test', type=lambda x: str(x).lower()=='true', default=False, 
                        help='log the testing error into files')
    parser.add_argument('--lr', type=float, default=1e-3, 
                        help="learning rate for the optimizer")
    parser.add_argument('--latent-dim', type=int, default=200,
                        help="latent dimension")
    parser.add_argument('--categorical-dim', type=int, default=2,
                        help="categorical dimension")
    parser.add_argument('--optim', type=str, default='radam',
                        help="adam, radam")
    parser.add_argument('--activation', type=str, default='leakyrelu',
                        help="relu, leakyrelu")
    parser.add_argument('--number-of-samples', type=int, default=2,
                        help="number of samples for each input image")
    parser.add_argument('-s', '--gradient-estimate-sample', type=int, default=0,
                        help="number of samples used to estimate gradient bias (default 0: not estimate)")

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    print(args)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    
    args.dataset = args.dataset.lower()
    if args.dataset == 'mnist':
        train_dataset = [di for di in datasets.MNIST('./data/MNIST/', download=True, transform=transforms.ToTensor(), train=True)]
        valid_dataset = train_dataset[50000:]
        train_dataset = train_dataset[:50000]
        test_dataset = datasets.MNIST('./data/MNIST/', download=True, transform=transforms.ToTensor(), train=False)
    elif args.dataset == 'fashionmnist':
        train_dataset = [di for di in datasets.FashionMNIST('./data/FashionMNIST/', download=True, transform=transforms.ToTensor(), train=True)]
        valid_dataset = train_dataset[50000:]
        train_dataset = train_dataset[:50000]
        test_dataset = datasets.FashionMNIST('./data/FashionMNIST/', download=True, transform=transforms.ToTensor(), train=False)
    elif args.dataset == 'omniglot':
        import scipy.io 
        with open('omni_chardata.mat', 'rb') as fin:
            omni_raw = scipy.io.loadmat(fin)
        train_dataset = [
            (
                torch.Tensor(omni_raw["data"].T[i]).view(1, 28, 28), 
                0
            ) for i in range(omni_raw["data"].T.shape[0])
        ]
        valid_dataset = train_dataset[-1345:]
        train_dataset = train_dataset[:-1345]
        test_dataset = [
            (
                torch.Tensor(omni_raw["testdata"].T[i]).view(1, 28, 28), 
                0
            ) for i in range(omni_raw["testdata"].T.shape[0])
        ]
    else:
        print(f'dataset {args.dataset} is not supported!')
    
    input_size=train_dataset[0][0].numel()
    
    train_loader = torch.utils.data.DataLoader(
        [dynamic_binalization(*di) for di in train_dataset],
        batch_size=args.batch_size, shuffle=True, **kwargs)
    
    valid_loader = torch.utils.data.DataLoader(
        [dynamic_binalization(*di) for di in valid_dataset],
        batch_size=args.batch_size, shuffle=True, **kwargs)
    
    test_loader = torch.utils.data.DataLoader(
        [dynamic_binalization(*di) for di in test_dataset],
        batch_size=args.batch_size, shuffle=True, **kwargs)

    model = VAE(
        latent_dim = args.latent_dim, 
        categorical_dim = args.categorical_dim, 
        temperature = args.temperature, 
        hard = args.hard.lower() == 'true', 
        method = args.method, 
        activation=args.activation,
        input_size=input_size
    )
        
    if args.cuda:
        model.cuda()
        
    if args.optim.lower() == 'adam':
        optimizer_class = optim.Adam
    else:
        assert args.optim.lower() == 'radam'
        optimizer_class = optim.RAdam
        
    optimizer_0 = optimizer_class(model.parameters(), lr=args.lr)
    

    total_updates = 0
    epoch = 0
    try:
        while True:
            epoch += 1
            # model.temperature = 1 + (args.temperature - 1.) * (1 - (epoch - 1)/args.epochs)
            
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
                    
                bce, kld = 0, 0
                for _ in range(args.number_of_samples):
                    bce_i, kld_i, _, qy = model(data)
                    bce += bce_i
                    kld += kld_i
                bce /= args.number_of_samples
                kld /= args.number_of_samples
                loss = bce + kld
                optimizer_0.zero_grad()
                    
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), .1)
                optimizer_0.step()
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
            for i, (data, _) in enumerate(valid_loader):
                data = data.view(data.size(0), -1)
                if args.cuda:
                    data = data.cuda()
                bce, kld, _, __ = model(data)
                test_loss += (bce + kld).item() * len(data)
                test_bce += bce.item() * len(data)
                test_kld += kld.item() * len(data)

            print('====> Valid set loss: {:.6f} \t BCE: {:.6f} \t KLD: {:.6f}'.format(
                test_loss / len(valid_loader.dataset), 
                test_bce / len(valid_loader.dataset),
                test_kld / len(valid_loader.dataset)
            ))
            
            test_loss, test_bce, test_kld = 0, 0, 0
            temp = args.temperature
            for i, (data, _) in enumerate(test_loader):
                data = data.view(data.size(0), -1)
                if args.cuda:
                    data = data.cuda()
                bce, kld, _, __ = model(data)
                test_loss += (bce + kld).item() * len(data)
                test_bce += bce.item() * len(data)
                test_kld += kld.item() * len(data)

            print('====> Test set loss: {:.6f} \t BCE: {:.6f} \t KLD: {:.6f}'.format(
                test_loss / len(test_loader.dataset), 
                test_bce / len(test_loader.dataset),
                test_kld / len(test_loader.dataset)
            ))
            
            ### MISC ############

        if args.save_to:
            torch.save(model.state_dict(), args.save_to)
    except Exception as e: 
        print(e)
        print("Error in Training, Failed")
        