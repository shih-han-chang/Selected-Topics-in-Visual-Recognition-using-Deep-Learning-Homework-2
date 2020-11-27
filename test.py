from __future__ import print_function
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
#from data import VOC_ROOT, VOC_CLASSES as labelmap
from PIL import Image
#from data import VOCAnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
from data_ssd import BaseTransform
import torch.utils.data as data
from ssd import build_ssd
from torch.utils.data.dataset import Dataset
import torchvision.datasets as datasets
import cv2
import json
import time
import collections

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/ssd300_svhn_50000.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.6, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--test_root', default='test/test', help='Location of VOC root directory')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

def test_net(save_folder, net, cuda, root, transform, thresh):
    # dump predictions and assoc. ground truth to text file for now
    filename = save_folder+'out.json'
    fileList = os.listdir(root)
    fileList.sort(key=lambda x:int(x[:-4]))
    num_images = len(fileList)
    res = []
    for i in range(num_images):
        #print('Testing image {:d}/{:d}....'.format(i+1, num_images))
        img = cv2.imread(os.path.join(root,fileList[i]))
        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))

        if cuda:
            x = x.cuda()

        y = net(x)      # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])
        pred_num = 0
        coords = []
        score = []
        label_name = []
        res_dict = collections.OrderedDict()
        for k in range(detections.size(1)):
            j = 0
            while detections[0, k, j, 0] >= 0.25:
                pt = (detections[0, k, j, 1:]*scale).cpu().numpy()
                coords.append([str(round(pt[1],1)), str(round(pt[0],1)), str(round(pt[3],1)), str(round(pt[2],1))])
                s = (detections[0, k, j, 0]).cpu().numpy()
                #print(s)
                score.append(str(s))
                label_name.append(str(k))
                pred_num += 1
                j += 1
        res_dict['bbox'] = coords
        res_dict['score'] = score
        res_dict['label'] = label_name
        #res_dict = {'name':[fileList[i]],'bbox':coords,'score':[score],'label':[label_name]}
        res.append(res_dict)
        
    res = json.dumps(res) 
    with open(filename, 'a') as f:
        f.write(res)
    #    for i in range(detections.size(1)):
    #        j = 0
    #        while detections[0, i, j, 0] >= 0.5:
    #            if pred_num == 0:
    #                with open(filename, mode='a') as f:
    #                    f.write('PREDICTIONS: '+'\n')
    #            score = detections[0, i, j, 0]
    #            label_name = str(i)
    #            pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
    #            coords = (pt[0], pt[1], pt[2], pt[3])
    #            pred_num += 1
    #            with open(filename, mode='a') as f:
    #                f.write(str(pred_num)+' label: '+label_name+' score: ' +
    #                        str(score) + ' '+' || '.join(str(c) for c in coords) + '\n')
    #            j += 1


def test_svhn():
    # load net
    num_classes = 11 # +1 background
    net = build_ssd('test', 300, num_classes) # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data
    #testset = image_dataset_test(args.test_root, None)
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net, args.cuda, args.test_root,
             #BaseTransform(net.size, (104, 117, 123)),
             BaseTransform(net.size, (88, 95, 98)),
             thresh=args.visual_threshold)

if __name__ == '__main__':
    test_svhn()
