import os
import sys
import yaml
import numpy as np
import torch
import utils
import codecs
import json
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.autograd import Variable
from utils import Genotype


FILE_ABSOLUTE_PATH = os.path.abspath(__file__)
cell_images_path = os.path.dirname(FILE_ABSOLUTE_PATH)
project_path = os.path.dirname(cell_images_path)
robustdarts_src_evaluation_path = os.path.join(project_path, 'RobustDARTS', 'src', 'evaluation')
sys.path.append(robustdarts_src_evaluation_path)
print(sys.path)

from model import NetworkImageNet as Network
from malaria_dataset import MalariaImageLabelDataset

TORCH_VERSION = torch.__version__


parser = argparse.ArgumentParser("imagenet")
parser.add_argument('--data', type=str, default='../data/imagenet/', help='location of the data corpus')
parser.add_argument('--space', type=str, default='s1', help='space index')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
parser.add_argument('--search_wd', type=float, default=3e-4, help='weight decay used during search')
parser.add_argument('--search_dp', type=float, default=0.2, help='drop path probability used during search')

parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--init_channels', type=int, default=48, help='num of init channels')
parser.add_argument('--layers', type=int, default=14, help='total number of layers')
parser.add_argument('--model_path', type=str, default='EXP/model.pt', help='path of pretrained model')

parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')

parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')

parser.add_argument('--search_task_id', type=int, default=1, help='SLURM_ARRAY_TASK_ID number during search')
parser.add_argument('--task_id', type=int, default=1, help='SLURM_ARRAY_TASK_ID number')

parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')

# logging options
parser.add_argument('--debug', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--save', type=str, default='experiments/eval_logs', help='log directory name')
parser.add_argument('--archs_config_file', type=str, default='./experiments/search_logs/results_arch.yaml',
                    help='search logs directory')
parser.add_argument('--results_test', type=str, default='results_perf', help='filename where to write test errors')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')

parser.add_argument('--archs_config_file', type=str, default='../RobustDARTS/experiments/search_logs/results_arch.yaml',
                    help='search logs directory')

args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

CLASSES = 2


if TORCH_VERSION.startswith('1'):
    device = torch.device('cuda:{}'.format(args.gpu))

def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  # if args.dataset == 'dr-detection':
  #   configuration = '_'.join([args.space, 'cifar10'])
  # else:
  #   configuration = '_'.join([args.space, args.dataset])
  # settings\
  #   = '_'.join([str(args.search_dp), str(args.search_wd)])
  # with open(args.archs_config_file, 'r') as f:
  #   cfg = yaml.load(f, Loader=yaml.Loader)
  #   arch = dict(cfg)[configuration][settings][args.search_task_id]
  #
  # print(arch)
  genotype = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])
  model = Network(args.init_channels, 1000, args.layers, args.auxiliary, genotype)
  model = model.cuda()
  model.load_state_dict(torch.load(args.model_path, map_location='cuda:0')['state_dict'])
  model = nn.Sequential(
    model,
    nn.Linear(1000,1),
    nn.ReLU()
  )

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))



  train_transform = transforms.Compose([
      transforms.Resize(100),
      transforms.RandomCrop(64),  # 224
      transforms.RandomHorizontalFlip(),
      transforms.RandomVerticalFlip(),
      transforms.ToTensor(),
  ])
  train_data = MalariaImageLabelDataset(transform=train_transform, train=True)
  valid_data = MalariaImageLabelDataset(transform=train_transform, train=False)

  train_queue = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size,
    shuffle=True, pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
    valid_data, batch_size=args.batch_size,
    shuffle=False, pin_memory=True, num_workers=2)

  optimizer = torch.optim.SGD(
    model.parameters(),
    args.learning_rate,
    momentum=args.momentum,
    weight_decay=args.weight_decay
  )

  scheduler = CosineAnnealingLR(
    optimizer, float(args.epochs))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()

  errors_dict = {'train_acc': [], 'train_loss': [], 'valid_acc': [],
                'valid_loss': []}

  for epoch in range(args.epochs):
    scheduler.step()
    logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    # training
    train_acc, train_obj = train(train_queue, model, criterion, optimizer)
    logging.info('train_acc %f', train_acc)

    # evaluation
    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc %f', valid_acc)

    # update the errors dictionary
    errors_dict['train_acc'].append(100 - train_acc)
    errors_dict['train_loss'].append(train_obj)
    errors_dict['valid_acc'].append(100 - valid_acc)
    errors_dict['valid_loss'].append(valid_obj)

  with codecs.open(os.path.join(args.save,
                                'errors_{}_{}.json'.format(args.search_task_id, args.task_id)),
                   'w', encoding='utf-8') as file:
    json.dump(errors_dict, file, separators=(',', ':'))

  utils.write_yaml_results_eval(args, args.results_test, 100-valid_acc)

  model.drop_path_prob = args.drop_path_prob
  valid_acc_top1, valid_acc_top5, valid_obj = infer(valid_queue, model, criterion)
  logging.info('valid_acc_top1 %f', valid_acc_top1)
  logging.info('valid_acc_top5 %f', valid_acc_top5)

def train(train_queue, model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.train()

  for step, input_target in enumerate(train_queue):

    if args.dataset == 'dr-detection':
        input = input_target['image']
        target = input_target['label']
    else:
        input = input_target[0]
        target = input_target[1]

    if TORCH_VERSION in ['1.0.1', '1.1.0']:
      input = input.to(device)
      target = target.to(device)
    else:
      input = Variable(input).cuda()
      target = Variable(target).cuda()

    optimizer.zero_grad()
    logits, logits_aux = model(input)
    loss = criterion(logits, target)
    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight*loss_aux
    loss.backward()

    if TORCH_VERSION.startswith('1'):
      nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    else:
      nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    if args.dataset == 'malaria':
      prec1 = utils.accuracy(logits, target)
      prec1 = prec1[0]
      n = input.size(0)
      if TORCH_VERSION.startswith('1'):
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
      else:
        objs.update(loss.data[0], n)
        top1.update(prec1.data[0], n)
    else:
      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      if TORCH_VERSION.startswith('1'):
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
      else:
        objs.update(loss.data[0], n)
        top1.update(prec1.data[0], n)
        top5.update(prec5.data[0], n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
      if args.debug:
        break

  return top1.avg, objs.avg

def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda()

    logits, _ = model(input)
    loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data[0], n)
    top1.update(prec1.data[0], n)
    top5.update(prec5.data[0], n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, top5.avg, objs.avg


if __name__ == '__main__':
  main()