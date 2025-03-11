import os
import time
import argparse
import random
import warnings
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torchvision import transforms
from timm.scheduler import CosineLRScheduler

from utils import accuracy, AverageMeter, ProgressMeter, Summary, save_checkpoint, Cutmix_Mixup
import networks as models


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
# Parse input arguments
parser = argparse.ArgumentParser(description='Single Fractal Training',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# loading
parser.add_argument('--datadir', default='/data/users/cugwu/ad_data/iccv2025/1p-fractals/', type=str,
                    help='path to dataset or metadata.csv')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--outdir', type=str, default='./outputs')
parser.add_argument('--store_name', type=str, default="test")
# training
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training.')
parser.add_argument('--epochs', type=int, default=8, help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=10, help='input batch size for training')
parser.add_argument('--accumulation_steps', default=1, type=int)
parser.add_argument('--base_lr', type=float, default=0.1, help='learning rate for a single GPU')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument("--weight_decay", default=1e-4, type = float, help="weight decay")
parser.add_argument('--warmup_epochs', type=float, default=10, help='number of warmup epochs')
parser.add_argument('--eval_step', type=int, default=100, help='print result every N batch')
parser.add_argument('--print_freq', type=float, default=100, help='print result every N batch')
parser.add_argument('--evaluate', dest='evaluate', action='store_true',help='evaluate model on validation set')
parser.add_argument('--workers', default=1, type=int, help='number of data loading workers (default: 8)')
# model & data
parser.add_argument('--num_class', type=int, default=1000)
parser.add_argument('--arch', default='resnet50', choices=model_names, type=str,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet50)'),
parser.add_argument('--crop_size', type=int, default=224, help='size of the image')
parser.add_argument('--alpha', type=float, default=1.0, help='cutmix-mixup alpha parameter')
parser.add_argument('--mix_prob', default=0.8, type=lambda x: 0.0 <= float(x) <= 1.0, metavar='0.0-1.0',
                    help='cutmix mixup probabilities')
args = parser.parse_args()

start_epoch = 0
best_acc = 0

def main():
    global start_epoch
    global best_acc

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5),

    ])
    test_transform = transforms.Compose([
        transforms.Resize((args.crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # Initialize MixUp and CutMix transforms
    augmentations = Cutmix_Mixup(args.alpha)

    train_set = ImageFolder(root=args.datadir, transform=train_transform)
    test_set =  ImageFolder(root=args.datadir, transform=test_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=False,
                                               num_workers=args.workers)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, drop_last=False,
                                              num_workers=args.workers)

    # Model creation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.__dict__[args.arch](num_classes=args.num_class)
    model.to(device)

    # Define loss function
    loss_fn = nn.CrossEntropyLoss().to(device)

    # Define the SGD optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = CosineLRScheduler(optimizer,
                                  t_initial=args.epochs,
                                  lr_min=1.0e-5,
                                  warmup_t=5,
                                  warmup_lr_init=1.0e-6,
                                  cycle_limit=1,
                                  t_in_epochs=True)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizers'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    tf_writer = None
    filename = os.path.join(args.outdir, args.store_name, 'logs')
    os.makedirs(filename, exist_ok=True)
    with open(os.path.join(filename, 'args.txt'), 'w') as f: f.write(str(args))
    tf_writer = SummaryWriter(log_dir=filename)

    if args.evaluate:
        _ = test(model, test_loader, start_epoch, loss_fn, device, tf_writer, args)
        return

    for epoch in range(start_epoch, args.epochs):
        train(model, optimizer, train_loader, epoch, loss_fn, device, tf_writer, augmentations)
        torch.cuda.empty_cache()
        scheduler.step(epoch)

        if (epoch + 1) % args.eval_step == 0:
            acc1 = test(model, test_loader, epoch, loss_fn, device, tf_writer, args)
            torch.cuda.empty_cache()

            # Remember the best loss and save checkpoint
            is_best = acc1 > best_acc
            best_acc = max(acc1, best_acc)

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizers': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, is_best, args)

    print(args.arch, 'best accuracy:', best_acc)



def train(model, optimizer, train_loader, epoch, loss_fn, device, tf_writer, augmentations):
    batch_time = AverageMeter('Batch Time', ':6.3f')
    data_time = AverageMeter('Data Loading', ':6.3f')
    losses = AverageMeter('Training Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    end = time.time()
    cutmix_mixup = augmentations
    for i, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        labels = labels.to(device)
        images = images.to(device)

        # images = freq_swap(images)
        p = random.uniform(0, 1)
        if p < args.mix_prob:
            mixed_images, labels_a, labels_b, lam = cutmix_mixup(images, labels)

            outputs = model(mixed_images)
            loss = loss_fn(outputs, labels_a) * lam + loss_fn(outputs, labels_b) * (1. - lam)
        else:
            outputs = model(images)
            loss = loss_fn(outputs, labels)

        acc1 = accuracy(outputs, labels, topk=(1,))

        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))

        optimizer.zero_grad()
        loss.backward()
        if (i + 1) % args.accumulation_steps == 0:
            optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)
            tf_writer.add_scalar('loss/train', losses.avg, epoch * len(train_loader) + i)
            tf_writer.add_scalar('acc/train_top1', top1.avg, epoch * len(train_loader) + i)

    return


def test(model, test_loader, epoch, loss_fn, device, tf_writer, args):
    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, labels) in enumerate(loader):
                i = base_progress + i

                labels = labels.to(device)
                images = images.to(device)

                outputs = model(images)
                # loss
                loss = loss_fn(outputs, labels)
                acc1 = accuracy(outputs, labels, topk=(1,))

                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0].item(), images.size(0))

                batch_time.update(time.time() - end, images.size(0))
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

        if tf_writer is not None:
            tf_writer.add_scalar('loss/test', losses.avg, epoch)
            tf_writer.add_scalar('acc/test_top1', top1.avg, epoch)


    batch_time = AverageMeter('Images/sec', ':6.3f', Summary.NONE)
    losses = AverageMeter('Validation Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(test_loader) ,
        [batch_time, losses, top1],
        prefix='Test: ')

    model.eval()
    run_validate(test_loader)

    return top1.avg


if __name__ == '__main__':
    main()