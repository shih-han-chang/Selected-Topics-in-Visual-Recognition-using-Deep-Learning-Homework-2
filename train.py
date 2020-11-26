from data_ssd import *
from utils_ssd.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import random


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(
# train_set = parser.add_mutually_exclusive_group()
parser.add_argument('-p', dest = 'pretrain',    default='vgg16_reducedfc.pth',    help='Model pretrain weights')
parser.add_argument('-b', dest = 'batch_size',  default=16,        type=int,      help='Batch size for training')
parser.add_argument('-r', dest = 'resume',      default=None,      type=str,      help='Checkpoint state_dict file to resume training from')
parser.add_argument('-m', dest = 'max_iter',    default=10001,    type=int,      help='Max training iteration')
parser.add_argument('-s', dest = 'save_folder', default='weights/',               help='Directory for saving checkpoint models')
parser.add_argument('-d', dest = 'disp_interval', default=20,      type=int,  help='number of iterations to display')
# parser.add_argument('-v', dest = 'visdom',      default=False,     type=str2bool, help='[Not supported] Use visdom for loss visualization')
args = parser.parse_args() 

# Training settings
Define_visualize_training_data = False
save_iteration= 2000
initial_lr = 1e-4
lr_step = [2000,6000,8000]

torch.set_default_tensor_type('torch.cuda.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

# data.py
import torch.utils.data as data
import numpy as np
import cv2
import os
import pandas as pd
import h5py

class image_dataset(data.Dataset):
    def __init__(self, mat_file, img_folder, transform=None):
        self.hdf5_data = h5py.File(mat_file,'r')
        self.root = img_folder
        self.transform = transform 
    def __getitem__(self, index):
        img, target, hight, width = self.img_data_constructor(index)
        return img, target
    def __len__(self):
        return len(self.hdf5_data['/digitStruct/name'])

    def get_name(self, index):
      name = self.hdf5_data['/digitStruct/name']
      return ''.join([chr(v[0]) for v in self.hdf5_data[name[index][0]].value])

    def get_bbox(self, index, w, h):
      attrs = {}
      item = self.hdf5_data['digitStruct']['bbox'][index].item()
      res = []
      for key in ['label', 'left', 'top', 'width', 'height']:
        attr = self.hdf5_data[item][key]
        values = [self.hdf5_data[attr.value[i].item()].value[0][0]
                  for i in range(len(attr))] if len(attr) > 1 else [attr.value[0][0]]
        attrs[key] = values
      for i in range(len(attrs['label'])):
        bbox = []
        bbox.append(attrs['left'][i]/w)
        bbox.append(attrs['top'][i]/h)
        bbox.append((attrs['left'][i]+attrs['width'][i])/w)
        bbox.append((attrs['top'][i]+attrs['height'][i])/h)
        bbox.append(attrs['label'][i])
        res += [bbox] 
      return res

    def img_data_constructor(self, index):
      bbox_df = []
      #row_dict = {}
      img_name = self.get_name(index)
      img = cv2.imread(os.path.join(self.root, img_name)) # Read img
      height, width, _ = img.shape
      target = self.get_bbox(index, height, width)
      row_dict_total = {'img_name':[img_name],'img_height':[img.shape[0]],'img_width':[img.shape[1]],'img':[img],'target':[target]}
      #target = row_dict['bbox']
      #print(target)
      if self.transform is not None:
        target = np.array(target)
        img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
        # to rgb
        img = img[:, :, (2, 1, 0)]
        target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
      return torch.from_numpy(img).permute(2, 0, 1), target, height, width

def train():
    print("torch version:",torch.__version__)
    #==================#
    #  DataSet Config  #
    #==================#
    cfg = svhn
    #dataset_root = IVS_ROOT
    #image_list = 'train'
    dataset = image_dataset(mat_file="/content/drive/MyDrive/詩涵的作業/Homework2/train/digitStruct.mat",
						  img_folder="/content/drive/MyDrive/詩涵的作業/Homework2/train", 
                          transform=SSDAugmentation(300,(104, 117, 123)) )
                           # transform=None)
    data_loader = data.DataLoader(dataset, args.batch_size,
                              num_workers=4,
                              shuffle=True, collate_fn=detection_collate,
                              pin_memory=True)
    if Define_visualize_training_data:
        print(random.randint(1,9000))
        for i in ([random.randint(1,9000) for i in range(100)]):
            img = dataset.pull_item(i)
            img = img[0].numpy()
            img = np.swapaxes(img,0,2)
            img = np.swapaxes(img,0,1)
            cv2.imshow('Image',img)
            cv2.waitKey(0)
    '''
    if args.visdom:
        import visdom
        viz = visdom.Visdom()
    '''
    
    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    net = ssd_net
    net = torch.nn.DataParallel(ssd_net)
    cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    else:
        vgg_weights = torch.load(args.save_folder + args.pretrain)
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)
    net = net.cuda()
    
    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=0.9, weight_decay=5e-4)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5, False, True)

    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading the dataset...')
    epoch_size1 = len(dataset) / args.batch_size 
    print('Training SSD on:', dataset.name)

    '''
    if args.visdom:
        vis_title = 'SSD.PyTorch on ' + dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)
    '''
    
    # create batch iterator
    step_index = 0
    batch_iterator = iter(data_loader)
    lr = initial_lr
    for iteration in range(args.max_iter):
        '''
        if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
            update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None,
                            'append', epoch_size)
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            epoch += 1
        '''
        if iteration in lr_step:
            step_index += 1
            gamma = 0.1
            lr = adjust_learning_rate(lr, optimizer, gamma, step_index)

        # load train data
        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)

        images = Variable(images.cuda())
        targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
        
        # forward
        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        loc_loss += loss_l.data[0]
        conf_loss += loss_c.data[0]

        if iteration % args.disp_interval == 0:
            print('%s ' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))),
            print('iter: %s , Loss: %.4f , lr = %.3e ' % (repr(iteration), loss.data[0], lr))
            
        if iteration != 0 and iteration % save_iteration == 0:
            saves_weights = args.save_folder+'ssd300_IVS_ta_v2_voc_' + repr(iteration) + '.pth'
            print('Saving: ',saves_weights)
            torch.save(ssd_net.state_dict(),saves_weights)
        '''
        if args.visdom:
            update_vis_plot(iteration, loss_l.data[0], loss_c.data[0], iter_plot, epoch_plot, 'append')
        '''

def adjust_learning_rate(initial_lr,optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = initial_lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


if __name__ == '__main__':
    train()
