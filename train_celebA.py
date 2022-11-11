from __future__ import print_function
from ast import While
from tokenize import group
from sklearn import metrics
import pandas as pd
from inspect import Parameter
from torch.utils.data import Dataset, DataLoader
from Knn_sim import Knn_sim
from math import ceil
import torchvision
import copy
from torch.nn.utils import prune
from torch.cuda.amp import autocast, GradScaler
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
from os import path
import argparse
import numpy as np
from sklearn.mixture import GaussianMixture
from dataset_celebA import prepare_data

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=128,
                    type=int, help='train batchsize')
parser.add_argument('--knn', default=100, type=int, help=' ')
parser.add_argument('--epochs', default=20, type=int, help='epochs')
parser.add_argument('--warm_up', default=5, type=int, help='warm epochs')
parser.add_argument('--lr', '--learning_rate', default=0.0001,
                    type=float, help='initial learning rate')
parser.add_argument('--weight_decay', default=0.0001, type=float)
parser.add_argument('--save_name', type=str, default='robust')
parser.add_argument('--top', default=1.0, type=float, help='top')
parser.add_argument('--r', default=0, type=float, help='noise ratio')
parser.add_argument('--GMMtol', default=0.001, type=float)
parser.add_argument('--GMMreg_covar', default=1e-3, type=float)
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--add', default=1, type=int)
parser.add_argument('--lambda_l', default=1, type=float,
                    help='weight for supervised loss')
parser.add_argument('--T', default=1, type=int)
parser.add_argument('--num_class', default=2, type=int)
parser.add_argument('--labelconf', default='ce', type=str)
parser.add_argument('--unshifted_val', action='store_true', default=False)


parser.add_argument('--root_dir', default='../data/celebA/', type=str)
parser.add_argument('--shift_type', default='confounder', type=str)
parser.add_argument('--dataset', default='celebA', type=str)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--augment_data', action='store_true', default=False)
parser.add_argument('--reweight_groups', action='store_true', default=False)
args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
cudnn.benchmark = True

N = 162770
parameters = {}

for i in ['lr', 'weight_decay', 'GMMtol', 'GMMreg_covar']:
    parameters[i] = eval(f'args.{i}')


class prettyfloat(float):
    def __repr__(self):
        return "%0.2f" % self


def save_model(save_name, save_path, net1, net2, optimizer1, optimizer2):
    save_filename = os.path.join(save_path, save_name)
    torch.save({'net1': net1.state_dict(),
                'net2': net2.state_dict(),
                'optimizer1': optimizer1.state_dict(),
                'optimizer2': optimizer2.state_dict(),
                }, save_filename)
    print(f"model saved: {save_filename}")


def entropy(p):
    return - torch.sum(p * torch.log(p), axis=-1)


class SoftCELoss_topk(object):
    def __call__(self, outputs, targets):
        loss = -torch.sum(F.log_softmax(outputs, dim=1) * targets, dim=1)
        vals, idx = loss.topk(int(args.top*loss.shape[0]))
        Lx = torch.mean(loss[idx])
        return Lx


class SoftCELoss(object):
    def __call__(self, outputs, targets, weight=None):
        if weight is not None:

            Lx = -torch.sum(torch.sum(F.log_softmax(outputs,
                            dim=1) * targets, dim=1)*weight)
        else:
            Lx = - \
                torch.mean(torch.sum(F.log_softmax(
                    outputs, dim=1) * targets, dim=1))
        return Lx


def load_model(load_path, net1, net2, optimizer1, optimizer2):
    checkpoint = torch.load(load_path)
    for key in checkpoint.keys():
        if 'net1' in key:
            net1.load_state_dict(checkpoint[key])
        elif 'net2' in key:
            net2.load_state_dict(checkpoint[key])
        elif key == 'optimizer1':
            optimizer1.load_state_dict(checkpoint[key])
        elif key == 'optimizer2':
            optimizer2.load_state_dict(checkpoint[key])
        print(f"Check Point Loading: {key} is LOADED")


def train(epoch, net1, net2, ema_model, optimizer, labeled_trainloader, mode='PL'):
    net1.train()
    net2.eval()
    hard_label = False

    scaler = GradScaler()

    labeled_train_iter = iter(labeled_trainloader)
    I = 1

    while True:

        try:
            inputs, labels_x, _, F = labeled_train_iter.next()
            inputs_w, inputs_s = inputs
        except StopIteration:
            if I == 1:
                return
            else:
                labeled_train_iter = iter(labeled_trainloader)
                inputs, labels_x, _, F = labeled_train_iter.next()
                inputs_w, inputs_s = inputs
                I = I-1

        F = F.cuda(args.gpuid)

        batch_size = inputs_w.size(0)

        one_hot_x = torch.zeros(batch_size, args.num_class).scatter_(
            1, labels_x.view(-1, 1), 1)
        one_hot_x = one_hot_x.cuda(args.gpuid)

        inputs_w, inputs_s, labels_x = inputs_w.cuda(
            args.gpuid), inputs_s.cuda(args.gpuid), labels_x.cuda(args.gpuid)

        with torch.no_grad():
            outputs_1_w = net1(inputs_w)
            outputs_2_w = net2(inputs_w)

        with autocast():
            outputs_s = net1(inputs_s)
            w_x = F.view(-1, 1)
            targets_x = (torch.softmax(outputs_1_w, dim=1) +
                         torch.softmax(outputs_2_w, dim=1))/2

            targets_x = targets_x**args.T
            targets_x = targets_x/targets_x.sum(dim=1, keepdim=True)
            targets_x = targets_x.detach()

            targets_x = one_hot_x*w_x+(1-w_x)*targets_x
            Lx = softCELoss_topk(outputs_s, targets_x)
            loss = args.lambda_l*Lx

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


def warmup(epoch, net, ema_model, ema, optimizer, dataloader, updataema=False):
    net.train()
    scaler = GradScaler()
    total_loss = 0
    all_targets = []
    all_preds = []
    with autocast():
        for batch_idx, (inputs, labels, group_label) in enumerate(dataloader):

            inputs, labels = inputs.cuda(args.gpuid), labels.cuda(args.gpuid)

            outputs = net(inputs)

            loss = CEloss(outputs, labels)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds)
            all_targets.append(labels)

            L = loss

            optimizer.zero_grad()
            scaler.scale(L).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
    all_targets = torch.cat(all_targets, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    print((all_targets == all_preds).sum())
    print(f'Loss {total_loss}')


@torch.no_grad()
def test(epoch, net1, net2, test_loader):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    all_targets = []
    all_groups = []
    all_preds = []
    with torch.no_grad():
        for batch_idx, (inputs, targets, group_target) in enumerate(test_loader):
            inputs, targets = inputs.cuda(args.gpuid), targets.cuda(args.gpuid)
            group_target = group_target.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)
            outputs = torch.softmax(outputs1, dim=1) + \
                torch.softmax(outputs2, dim=1)
            _, predicted = torch.max(outputs, 1)
            all_preds.append(predicted)
            all_targets.append(targets)
            all_groups.append(group_target)
    all_targets = torch.cat(all_targets, dim=0)
    all_groups = torch.cat(all_groups, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    acc = (all_targets == all_preds).float().mean().item()*100
    group_acc = []
    group_num = []
    for i in range(4):
        t = all_groups == i
        group_acc.append(
            (all_targets[t] == all_preds[t]).float().mean().item()*100)
        group_num.append(t.sum().item())
    unshifted_acc = acc
    text = f'Epoch {epoch}, acc {acc:.2f}, {group_acc}, min: {min(group_acc)}'
    print(text)
    if args.unshifted_val:
        return unshifted_acc
    else:
        return acc


@torch.no_grad()
def eval_train(model, dataloader):
    gmm = GaussianMixture(n_components=2, max_iter=10,
                          tol=parameters['GMMtol'], reg_covar=parameters['GMMreg_covar'], warm_start=False)
    model.eval()
    losses = []
    all_group = []
    all_target = []

    with torch.no_grad():
        for batch_idx, (inputs, targets, group) in enumerate(dataloader):

            inputs, targets = inputs.cuda(), targets.cuda()
            feats, outputs = model(inputs, feat=True)
            all_group.append(group)
            all_target.append(targets)

            loss = CE(outputs, targets)

            losses.append(loss)

    all_target = torch.cat(all_target, dim=0)
    all_group = torch.cat(all_group, dim=0)
    losses = torch.cat(losses, dim=0).detach().cpu().numpy()
    losses = (losses-losses.min())/(losses.max()-losses.min())

    input_loss = losses.reshape(-1, 1)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss)
    prob = prob[:, gmm.means_.argmin()]
    return prob


@torch.no_grad()
def eval_train_twostage(model, dataloader, classwise=False):

    gmm = GaussianMixture(n_components=2, max_iter=10,
                          tol=parameters['GMMtol'], reg_covar=parameters['GMMreg_covar'], warm_start=False)
    model.eval()

    epoch_feats = []
    all_targets = []
    all_group = []

    with torch.no_grad():
        for batch_idx, (inputs, targets, group) in enumerate(dataloader):

            inputs, targets = inputs.cuda(), targets.cuda()
            feats, outputs = model(inputs, feat=True)
            feats = F.normalize(feats, dim=1)
            epoch_feats.append(feats)
            all_targets.append(targets)
            all_group.append(group)

    epoch_feats = torch.cat(epoch_feats, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_group = torch.cat(all_group, dim=0)
    rscl = Knn_sim(epoch_feats, all_targets, graph=args.knn, mode='knn')

    epoch_sim = []
    epoch_losses = []
    for i in range(ceil(N/1000.0)):

        loss = rscl(epoch_feats[i*1000:(i+1)*1000],
                    labels=all_targets[i*1000:(i+1)*1000], reduction=False)
        epoch_losses.append(loss)

    epoch_losses = torch.cat(epoch_losses, dim=0).cpu().numpy()

    input_loss = epoch_losses.reshape(-1, 1)

    gmm.fit(input_loss)

    if classwise:
        labels = all_targets.cpu().numpy()
        all_prob = np.zeros(N)
        for c in range(args.num_class):
            gmm_c = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4,
                                    weights_init=gmm.weights_, means_init=gmm.means_, precisions_init=gmm.precisions_)
            gmm_c.fit(input_loss[labels == c])
            prob = gmm_c.predict_proba(input_loss[labels == c])
            all_prob[labels == c] = prob[:, gmm_c.means_.argmin()]
        return all_prob

    else:
        prob = gmm.predict_proba(input_loss)
        prob = prob[:, gmm.means_.argmin()]
        return prob


@torch.no_grad()
def get_pred(model1, model2):
    model1.eval()
    model2.eval()

    predictions1 = []

    predictions2 = []

    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(val_loader):
            inputs, targets = inputs.cuda(args.gpuid), targets.cuda(args.gpuid)
            outputs1 = model1(inputs)
            outputs2 = model2(inputs)

            predictions1.append(torch.softmax(outputs1, dim=1))

            predictions2.append(torch.softmax(outputs2, dim=1))

    predictions1 = torch.cat(predictions1, dim=0).detach().cpu().numpy()

    predictions2 = torch.cat(predictions2, dim=0).detach().cpu().numpy()

    return predictions1, predictions2, (predictions1+predictions2)/2


class NegEntropy(object):
    def __call__(self, outputs):
        probs = torch.softmax(outputs, dim=-1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))


def create_model():
    model = torchvision.models.resnet50(pretrained=True)
    model.fc = nn.Linear(2048, args.num_class)

    class resnet50(nn.Module):
        def __init__(self):
            super(resnet50, self).__init__()
            self.encoder = nn.Sequential(*list(model.children())[:-1])
            self.fc = nn.Sequential(list(model.children())[-1])

        def forward(self, x, feat=False):
            feature = self.encoder(x).view(x.shape[0], -1)
            if feat:
                return feature, self.fc(feature)
            else:
                return self.fc(feature)
    model = resnet50()
    model = model.cuda()
    return model


if args.r == 0:
    all_data, all_data_split = prepare_data(args, train=True, r=None)
else:
    all_data, all_data_split = prepare_data(args, train=True, r=args.r)

print('| Building net')


net1 = create_model()
net2 = create_model()


ema_model1 = None
ema_model2 = None


train_data, val_data, test_data = all_data_split['train'], all_data_split['val'], all_data_split['test']
print('data prepared')
loader_kwargs = {'batch_size': args.batch_size,
                 'num_workers': 4, 'pin_memory': True}
train_loader = DataLoader(train_data, shuffle=True, **loader_kwargs)
train_loader_woshuffle = DataLoader(train_data, shuffle=False, **loader_kwargs)
val_loader = DataLoader(val_data, shuffle=False, **loader_kwargs)
if test_data is not None:
    test_loader = DataLoader(test_data, shuffle=False, **loader_kwargs)


def get_optim(model):
    return torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=parameters['lr'], momentum=0.9, weight_decay=parameters['weight_decay'])


optimizer1 = get_optim(net1)
optimizer2 = get_optim(net2)

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()


softCE = SoftCELoss()
softCELoss_topk = SoftCELoss_topk()
softCELoss_lossWeight = SoftCELoss_lossWeight()
best_acc = 0

P1 = []
P2 = []
W1 = []
W2 = []


def add(moving_window, new_element):
    if len(moving_window) == args.add:
        del moving_window[0]
        moving_window.append(new_element)
    else:
        moving_window.append(new_element)
    return moving_window


p_epoch = 0

acc = 0
best_acc = 0
test_acc = 0
for epoch in range(p_epoch, args.warm_up):
    warmup_trainloader = train_loader
    print('Warmup Net1')
    warmup(epoch, net1, ema_model1, None, optimizer1, warmup_trainloader)
    print('Warmup Net2')
    warmup(epoch, net2, ema_model2, None, optimizer2, warmup_trainloader)

    print('validation set ', end='')
    acc = test(epoch, net1, net2, val_loader)
    if best_acc < acc:
        best_acc = acc
        test_acc = test(epoch, net1, net2, test_loader)
        print('test acc:', test_acc)
        save_model(f"{args.save_name}_{args.dataset}_{args.r}_best",
                   "./checkpoint/", net1, net2, optimizer1, optimizer2)

save_model(f"{args.save_name}_{args.dataset}_{args.r}_latest",
           "./checkpoint/", net1, net2, optimizer1, optimizer2)

for epoch in range(p_epoch, args.epochs):
    lr = parameters['lr']
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr

    if args.labelconf == 'ce':
        F1 = eval_train(net1, train_loader_woshuffle)
        F2 = eval_train(net2, train_loader_woshuffle)
    elif args.labelconf == 'lv':
        F1 = eval_train_twostage(net1, train_loader_woshuffle)
        F2 = eval_train_twostage(net2, train_loader_woshuffle)
    elif args.labelconf == 'ERM':
        F1 = np.ones(N)
        F2 = np.ones(N)

    all_data.set_conf(F1)
    train(epoch, net2, net1, ema_model2, optimizer2, train_loader)
    all_data.set_conf(F2)
    train(epoch, net1, net2, ema_model1, optimizer1, train_loader)
    all_data.set_conf(None)

    print('validation set ', end='')
    acc = test(epoch, net1, net2, val_loader)
    if best_acc < acc:
        best_acc = acc
        print('test set ', end='')
        test_acc = test(epoch, net1, net2, test_loader)
        save_model(f"{args.save_name}_{args.dataset}_{args.r}_best",
                   "./checkpoint/", net1, net2, optimizer1, optimizer2)

    save_model(f"{args.save_name}_{args.dataset}_{args.r}_latest",
               "./checkpoint/", net1, net2, optimizer1, optimizer2)
print('best test auc', test_acc)
