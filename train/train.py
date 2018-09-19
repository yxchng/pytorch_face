import os
import argparse
import shutil
import time
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import pytorch_face.models as models
from pytorch_face.utils import AverageMeter
from pytorch_face.datasets import *
from pytorch_face.optims import SutskeverSGD
from pytorch_face.layers import *
from pytorch_face.transforms import *


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Face Training')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='use CUDA')
    parser.add_argument('-j', '--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--print-freq', '-p', default=100, type=int,
                        help='print frequency (default: 100)')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--epochs', default=24, type=int, 
                        help='number of total epochs to run')
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='path to data folder')
    parser.add_argument('--data_list', type=str, required=True,
                        help='path to data list')
    parser.add_argument('--result_dir', type=str, 
                        help='path to save trained models', default="./results")
    model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))
    parser.add_argument('--arch', '-a', default='sphereface20',
                        choices=["sphereface20", "sphereface64"],
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: sphereface20)')
    parser.add_argument('--batch-size', '-b', default=256, type=int,
                        help='mini-batch size (default: 256)')
    parser.add_argument('--optimizer', '-o', default="sgd", type=str,
                        choices=["sgd", "sutskever_sgd"],
                        help='training optimizer')
    parser.add_argument('--learning-rate', '--lr', dest='lr', type=float,
                        help='initial learning rate', default=0.1)
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--mean', default=[127.5, 127.5, 127.5], type=float, nargs="+",
                        help='mean')
    parser.add_argument('--std', default=[128, 128, 128], type=float, nargs="+",
                        help='variance')
    parser.add_argument('--step', default=[12, 18, 24], type=int, nargs="+",
                        help='step or multisteps for learning rate')
    parser.add_argument('--lr_multiplier', default=0.1, dest='lr_mult', type=float,
                        help='factor to multiply with learning rate when step is reached')
    parser.add_argument('--margin_type',  default="multiplicative_angular", type=str,
                        choices=["multiplicative_angular", "additive_cosine"],
                        help='type of margin linear layer to use')
    parser.add_argument('--margin', '-m', default=4, type=int,
                        help='margin')
    parser.add_argument('--weight_scale', default=1, type=float,
                        help='scale of weights')
    parser.add_argument('--feature_scale', default=None, type=float,
                        help='scale of features')
    parser.add_argument('--lambda_decay', default="exponential", type=str,
                        choices=["exponential", "linear"],
                        help='type of lambda decay function to use')
    args = parser.parse_args()
    if args.lambda_decay == "exponential":
        additional_options = argparse.ArgumentParser(parents=[parser], add_help=False)
        additional_options.add_argument('--base', default=10000, type=float,
                            help='base of decay function')
        additional_options.add_argument('--start_iter', default=0, type=float,
                            help='start iteration of decay function')
        additional_options.add_argument('--lambda_min', default=8, type=float,
                            help='minimum lambda of decay function')
        additional_options.add_argument('--gamma', default=0.12, type=float,
                            help='gamma of decay function')
        additional_options.add_argument('--power', default=1, type=float,
                            help='power of decay function')
        args = additional_options.parse_args()
    return args


def main():
    print("==> parsing training configurations")
    args = parse_args()
    for arg in vars(args):
        print(' - {0:<16}: {1}'.format(arg, getattr(args, arg)))

    print("==> reading training data")
    if args.data_list:
        dataset = ImageList(
            args.data_dir,
            args.data_list,
            Compose([
                Normalize(args.mean, args.std),
                RandomHorizontalFlip(),
                ToTensor(),
            ]))
    else:
        dataset = ImageFolder(
            args.data_dir,
            Compose([
                Normalize(args.mean, args.std),
                RandomHorizontalFlip(),
                ToTensor(),
            ]))
    print(' - {0:<16}: {1}'.format("num_classes", dataset.num_classes()))

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None)

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    if args.margin_type in ["multiplicative_angular", "additive_cosine"]:
        margin_parameters = {'margin': args.margin,
                             'weight_scale': args.weight_scale,
                             'feature_scale': args.feature_scale}

    print("==> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](num_classes=dataset.num_classes(),
                                       margin_type=args.margin_type,
                                       margin_parameters=margin_parameters,
                                       is_deploy=False)
                                       
    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    
    if args.lambda_decay == "exponential":
        lambda_func = ExponentialDecay(args.base, args.start_iter, args.lambda_min, args.gamma, args.power)

    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), args.lr,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay)
    elif args.optimizer == "sutskever_sgd":
        optimizer = SutskeverSGD(model.parameters(), args.lr,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay)

    if len(args.step) > 1:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.step, gamma=args.lr_mult)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=args.lr_mult)

    if args.resume:
        resume = os.path.join(args.results, args.resume)
        if os.path.isfile(args.resume):
            print("==> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("==> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("==> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    train(device, data_loader, model, lambda_func, criterion, optimizer, scheduler, args)

def train(device, data_loader, model, lambda_func, criterion, optimizer, scheduler, args):
    model.train()
    best_loss = None

    start_timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    save_dir = args.result_dir + "/" + args.arch + "/" + start_timestamp
    print("==> training results to be saved at '{0}'".format(save_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    print("==> training")
    for epoch in range(1, args.epochs+1):
        batch_time = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        top1_m = AverageMeter()
        top5_m = AverageMeter()
        losses = AverageMeter()

        scheduler.step()

        end = time.time()
        for i, (input, target) in enumerate(data_loader):
            input = input.to(device)
            target = target.to(device)
            
            output = model(input)

            if args.lambda_decay == "":
                _lambda = 0
            else:
                _lambda = lambda_func.next()

            # can make this as a function
            x_dot_wT, f_m = output
            # must clone else modify in place
            f = x_dot_wT.clone()
            batch_size = target.size(0)
            idxs = torch.arange(0, batch_size, dtype=torch.long)
            f[idxs, target] = ((_lambda * x_dot_wT[idxs, target]) + f_m[idxs, target]) / (1 + _lambda) 

            loss = criterion(f, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            prec1, prec5 = accuracy(x_dot_wT, target, topk=(1, 5))
            prec1_m, prec5_m = accuracy(f, target, topk=(1, 5))
            losses.update(loss.item())
            top1.update(prec1[0])
            top5.update(prec5[0])
            top1_m.update(prec1_m[0])
            top5_m.update(prec5_m[0])

            current_timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
            if i % args.print_freq == 0:
                print('[{0}] '
                      'Epoch: [{1}/{2}] | '
                      'Iter: [{3}/{4}] | '
                      'LR: {lr:.1e} | '
                      'Lambda: {_lambda:.2f} | '
                      'Time/Batch: {batch_time.avg:.3f} | '
                      'CurrentLoss: {current_loss:.4f} | '
                      'AverageLoss: {average_loss.avg:.4f} | '
                      'MarginPrec@1: {top1_m.avg:.3f}% | '
                      'Prec@1: {top1.avg:.3f}% | '
                      'Prec@5: {top5.avg:.3f}%'.format(current_timestamp,
                       epoch, args.epochs, i+1, len(data_loader), lr=optimizer.param_groups[0]['lr'], _lambda=_lambda, batch_time=batch_time,
                       current_loss=loss.item(), average_loss=losses, top1_m=top1_m, top5_m=top5_m, top1=top1, top5=top5))
        if best_loss is None:
            is_best = True
            best_loss = losses.avg
        else:
            is_best = losses.avg < best_loss
            best_loss = min(losses.avg, best_loss)
        save_checkpoint({
            'epoch': epoch,
            'iter': i,
            'arch': args.arch,
            'metamodel': {'num_classes': model.num_classes, 
                          'margin_type': model.margin_type,
                          'margin_parameters': model.margin_parameters},
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'loss': best_loss,
            'prec1': top1.avg,
            'prec5': top5.avg
        }, is_best, save_dir)

    return losses.avg

def save_checkpoint(state, is_best, save_dir):
    checkpoint_path = os.path.join(save_dir, 'checkpoint.pth.tar')
    torch.save(state, checkpoint_path)
    if is_best:
        best_path = os.path.join(save_dir, 'epoch_' + str(state['epoch']) + '.pth.tar')
        shutil.copyfile(checkpoint_path, best_path)

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        max_k = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(max_k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            num_correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(num_correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    main()
