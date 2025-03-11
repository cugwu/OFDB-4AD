import os
import numpy as np
import shutil
from PIL import Image
from enum import Enum
import random
import torch
from torch.utils.data import Dataset
import torch.distributed as dist
from torchvision import transforms as T

class MVTecDataset(Dataset):
    def __init__(self, dataset_path='/data/users/cugwu/mvtec', class_name='bottle', is_train=True,
                 resize=256, cropsize=224):
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize

        print(self.class_name)

        # self.mvtec_folder_path = os.path.join(root_path, 'mvtec_anomaly_detection')

        # download dataset if not exist
        # self.download()

        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()

        # set transforms
        self.transform_x = T.Compose([T.Resize(resize, Image.LANCZOS),
                                      T.CenterCrop(cropsize),
                                      T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])])
        self.transform_mask = T.Compose([T.Resize(resize, Image.NEAREST),
                                         T.CenterCrop(cropsize),
                                         T.ToTensor()])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]

        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)

        if y == 0:
            mask = torch.zeros([1, self.cropsize, self.cropsize])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)

        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.png')])
            x.extend(img_fpath_list)

            # load gt labels
            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                 for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask)


class Cutmix_Mixup:
    def __init__(self, alpha):
        self.alpha = alpha
    @staticmethod
    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __call__(self, image, target):
        p = random.uniform(0, 1)
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        rand_index = torch.randperm(image.size()[0]).to(image.device)
        if p < 0.5:  # cutmix
            mixed_image = image
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(mixed_image.size(), lam)
            mixed_image[:, :, bbx1:bbx2, bby1:bby2] = mixed_image[rand_index, :, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (mixed_image.size()[-1] * mixed_image.size()[-2]))
            target_a, target_b = target, target[rand_index]
        else:  # mixup
            mixed_image = lam * image + (1 - lam) * image[rand_index, :]
            target_a, target_b = target, target[rand_index]

        return mixed_image, target_a, target_b, lam


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, step, len_epoch, args):
    """Warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * float(step) / (args.warmup_epochs * len_epoch)

    elif args.coslr:
        nmax = len_epoch * args.epochs
        lr = args.lr * 0.5 * (np.cos(step / nmax * np.pi) + 1)
    else:
        decay = 0.1 ** (sum(epoch >= np.array(args.lr_steps)))
        lr = args.lr * decay

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state, is_best, args):
    if not os.path.exists(os.path.join(args.outdir, args.store_name, 'checkpoint')):
        os.makedirs(os.path.join(args.outdir, args.store_name, 'checkpoint'))
    filename = os.path.join(args.outdir, args.store_name, 'checkpoint', "checkpoint.pth.tar")
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))

def restore_compatible_weights(model, state):
    '''Restores weights from a saved checkpoint. Any layers that have aren't compatible,
    such as classifier layers that have been adjusted for a new set of classes, are not restored,
    but left randomly initialized.
    '''
    net = model
    net_state = net.state_dict()
    new_state = {}
    incompatible = []
    for k in state:
        if k.startswith('model.'):
            kk = k[6:]  # Remove 'model.' prefix
        elif k.startswith('module.'):
            kk = k[7:]  # Remove 'module.' prefix
        elif k.startswith('backbone.'):
            kk = k[9:]  # Remove 'backbone.' prefix
        else:
            kk = k  # No matching prefix, use original k
        kk = kk.replace("glconv", "ffc")
        if kk in net_state:
            if net_state[kk].shape == state[k].shape:
                new_state[kk] = state[k]
            else:
                incompatible.append(kk)
    missing, extra = net.load_state_dict(new_state, strict=False)
    missing = [x for x in missing if x not in set(incompatible)]

    print('Model weights loaded:')
    print(f'  Missing keys: {", ".join(missing)}')
    print(f'  Extra keys: {", ".join(extra)}')
    print(f'  Incompatible sizes: {", ".join(incompatible)}')






