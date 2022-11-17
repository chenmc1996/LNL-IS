from __future__ import print_function
from Knn_sim import Knn_sim
from torch.cuda.amp import autocast, GradScaler
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
from vgg import vgg19_bn
from sklearn.mixture import GaussianMixture
import dataloader_animal as dataloader

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize') 
parser.add_argument('--epochs', default=50000000, type=int, help='epochs') 
parser.add_argument('--warm_up', default=15, type=int, help='warm epochs') 
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--noise_mode',  default='sym')
parser.add_argument('--knn', default=5000, type=int, help=' ')
parser.add_argument('--save_name', type=str, default='test')
parser.add_argument('--mode', type=str, default='')
parser.add_argument('--lambda_u', default=1, type=float, help='weight for unsupervised loss')
parser.add_argument('--lambda_l', default=1, type=float, help='weight for supervised loss')
parser.add_argument('--lambda_p', default=2, type=float, help='weight for penalty')
parser.add_argument('--T', default=1, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--pl_threshold', default=0.9, type=float, help='pl threshold')
parser.add_argument('--nl_threshold', default=0.1, type=float, help='nl threshold')
parser.add_argument('--num_epochs', default=500, type=int)
parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='../data/animal', type=str, help='path to dataset')
parser.add_argument('--dataset', default='animal', type=str)
parser.add_argument('--resume', default='', type=str)
args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

class prettyfloat(float):
    def __repr__(self):
        return "%0.2f" % self

def save_model( save_name, save_path,net1,net2,optimizer1,optimizer2,epoch):
    save_filename = os.path.join(save_path, save_name)
    torch.save({'net1': net1.state_dict(),
		'net2': net2.state_dict(),
                'optimizer1': optimizer1.state_dict(),
                'optimizer2': optimizer2.state_dict(),
                'epoch': epoch,
                }, save_filename)
    print(f"model saved: {save_filename}")

def entropy(p):
    return - torch.sum(p * torch.log(p), axis=-1)

class SoftCELoss(object):
    def __call__(self, outputs, targets):
        probs= torch.softmax(outputs, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs, dim=1) * targets, dim=1))
        return Lx

class SoftCELoss_topk(object):
    def __call__(self, outputs, targets):
        loss=-torch.sum(F.log_softmax(outputs, dim=1) * targets, dim=1)
        vals, idx = loss.topk(int(0.80*loss.shape[0]))
        Lx = torch.mean(loss[idx])
        return Lx

class SoftCELoss_lossWeight(object):
    def __call__(self, outputs, targets):
        loss=-torch.sum(F.log_softmax(outputs, dim=1) * targets, dim=1)
        weight=loss.detach()*0.5
        weight=torch.softmax(weight,dim=0)
        Lx = torch.sum(weight*loss)
        return Lx

def load_model(load_path,net1,net2,optimizer1,optimizer2):
    checkpoint = torch.load(load_path)
    epoch=0
    for key in checkpoint.keys():
        if 'net1' in key:
            net1.load_state_dict(checkpoint[key])
        elif 'net2' in key:
            net2.load_state_dict(checkpoint[key])
        elif key == 'optimizer1':
            optimizer1.load_state_dict(checkpoint[key])
        elif key == 'optimizer2':
            optimizer2.load_state_dict(checkpoint[key])
        elif key == 'epoch':
            epoch=checkpoint[key]
            
        print(f"Check Point Loading: {key} is LOADED")
    return epoch

@torch.no_grad()
def _eval_model_update(eval_model,train_model,ema_m=0.999):
    """
    Momentum update of evaluation model (exponential moving average)
    """
    for param_train, param_eval in zip(train_model.parameters(), eval_model.parameters()):
        param_eval.copy_(param_eval * ema_m + param_train.detach() * (1-ema_m))
    
    for buffer_train, buffer_eval in zip(train_model.buffers(), eval_model.buffers()):
        buffer_eval.copy_(buffer_train)            

# Training
def train(epoch,net1,net2,optimizer,labeled_trainloader,mode='PL',num_iter = 400):
    net1.train()
    net2.eval() #fix one network and train the other
    hard_label=False


    scaler=GradScaler()

    labeled_train_iter = iter(labeled_trainloader)    
    I=2

    for batch_idx in range(num_iter):
        try:
            inputs, labels_x, w_x,_ = labeled_train_iter.next()
            inputs_w,inputs_s =inputs
        except StopIteration:
            if I==1:
                stats_log.write(f'\n')
                stats_log.flush()
                return 
            else:
                labeled_train_iter = iter(labeled_trainloader)    
                inputs, labels_x, w_x,_ = labeled_train_iter.next()
                inputs_w,inputs_s =inputs
                I=I-1

        w_x = w_x.view(-1,1).type(torch.FloatTensor) 
        w_x = w_x.cuda()

        batch_size = inputs_w.size(0)

        one_hot_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)
        one_hot_x = one_hot_x.cuda()
	

        inputs_w,inputs_s, labels_x = inputs_w.cuda(),inputs_s.cuda(), labels_x.cuda()

        with torch.no_grad():

            outputs_1_w=net1(inputs_w)
            outputs_2_w=net2(inputs_w)

        with autocast():
            outputs_s=net1(inputs_s)

            if hard_label:
                pass
            else:
                probs=torch.softmax(outputs_1_w,dim=-1)+torch.softmax(outputs_2_w,dim=-1)
                probs=probs/2

                probs_T = probs**args.T
                probs_T = probs_T  / probs_T.sum(dim=1, keepdim=True)
                probs_T = probs_T.detach()
                probs_T_x,probs_T_u=probs_T[:batch_size],probs_T[batch_size:]
                targets_x=one_hot_x*w_x+(1-w_x)*probs_T_x
                Lx=softCELoss_topk(outputs_s, targets_x)

            prior = torch.ones(args.num_class)/args.num_class
            prior = prior.cuda()
            pred_mean = torch.softmax(outputs_s, dim=1).mean(0)
            penalty = torch.sum(prior*torch.log(prior/pred_mean))

            loss = args.lambda_l*Lx+args.lambda_p*penalty

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


def warmup(epoch,net,eval_models,ema,optimizer,dataloader,updataema=False):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        outputs = net(inputs)               
        loss = CEloss(outputs, labels)      
        if args.noise_mode=='asym':  # penalize confident prediction for asymmetric noise
            penalty = conf_penalty(outputs)
            L = loss + penalty      
        elif args.noise_mode=='sym':   
            L = loss
        L.backward()  
        optimizer.step() 
        if updataema:
            for eval_model,ema in zip(eval_models[:3],emas):
                _eval_model_update(eval_model,net,ema)

@torch.no_grad()
def test(epoch,net1,net2,class_wise=False):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    all_preds=[]
    all_target=[]
    with torch.no_grad():
        for batch_idx, (inputs, targets,_) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)           
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)            

            all_preds.append(predicted)
            all_target.append(targets)
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                 
    acc = 100.*correct/total
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))  

    if class_wise:
        all_preds=torch.cat(all_preds,dim=0).cpu().numpy()
        all_target=torch.cat(all_target,dim=0).cpu().numpy()
        print(np.mean(all_preds==all_target))
        for c in range(args.num_class):
            print(np.mean(all_preds[all_target==c]==c))
        print(all_target.tolist())
        print(all_preds.tolist())

    stats_log.write('Epoch:%d   Accuracy:%.2f\n'%(epoch,acc))
    stats_log.flush()  
    return acc
    

@torch.no_grad()
def eval_train_twostage(model,prob,all_loss,all_preds,noise,gmm,obtain_confidence=True):
    model.eval()
    epoch_losses = torch.zeros(50000)    
    epoch_feats = torch.zeros(50000,4096).cuda()
    all_targets = torch.zeros(50000)

    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            feats, outputs = model(inputs,feat=True) 
            feats = F.normalize(feats, dim=1)
            for b in range(targets.size(0)):
                epoch_feats[index[b]]=feats[b]         
                all_targets[index[b]]=targets[b]         
    
    labels=all_targets.cpu().numpy()
    if obtain_confidence:
        rscl=Knn_sim(epoch_feats,all_targets,graph=args.knn,contrast_mode='knn')
        epoch_losses=[]
        for i in range(50):
            epoch_losses.append(rscl(epoch_feats[i*1000:(i+1)*1000],labels=all_targets[i*1000:(i+1)*1000],reduction=False)) 
        epoch_losses=torch.cat(epoch_losses,dim=0).cpu().numpy()

        all_loss.append(epoch_losses)

        input_loss = epoch_losses.reshape(-1,1)

        gmm.fit(input_loss)
        prob = gmm.predict_proba(input_loss) 
        prob = prob[:,gmm.means_.argmin()]         
        return prob,all_loss#,predictions
    else:
        return all_preds


class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=2)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

def create_model():
    model = vgg19_bn(num_classes=args.num_class, pretrained=False)
    model = model.cuda()
    return model

stats_log=open('./checkpoint/%s_%s_%.1f_%s'%(args.save_name,args.dataset,args.r,args.noise_mode)+'_stats.txt','w') 
loader = dataloader.animal_dataloader(root=args.data_path,batch_size=args.batch_size,num_workers=5)

print('| Building net')

net1 = create_model()
net2 = create_model()
test_loader = loader.run('test')
acc=test(-1,net1,net2)  

cudnn.benchmark = True

optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
softCE=SoftCELoss()
softCELoss_topk=SoftCELoss_topk()
softCELoss_lossWeight=SoftCELoss_lossWeight()
best_acc=0

all_loss_1=[]
all_loss_2=[]

if len(args.resume)!=0:
    p_epoch=load_model(args.resume,net1,net2,optimizer1,optimizer2)
    test_loader = loader.run('test')
    acc=test(p_epoch-1,net1,net2,class_wise=True)  
    # p_epoch=150
    exit()
else:
    p_epoch=0

for epoch in range(p_epoch,args.warm_up):   
    test_loader = loader.run('test')
    for param_group in optimizer1.param_groups:
        param_group['lr'] = args.lr*2
    for param_group in optimizer2.param_groups:
        param_group['lr'] = args.lr*2
    warmup_trainloader = loader.run('warmup')
    print('Warmup Net1')
    warmup(epoch,net1,None,None,optimizer1,warmup_trainloader)
    print('Warmup Net2')
    warmup(epoch,net2,None,None,optimizer2,warmup_trainloader)

    acc=test(epoch,net1,net2)
    
save_model(f"{args.save_name}_{args.dataset}_{args.r}_{args.noise_mode}_latest","./checkpoint/",net1,net2,optimizer1,optimizer2,0)

gmm4loss = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4,warm_start=False)

for epoch in range(p_epoch,args.num_epochs+1):   
    lr=args.lr
    if args.num_epochs-epoch<100 :
        lr /= 10 
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr    
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr    
        
    test_loader = loader.run('test')
    eval_loader = loader.run('eval_train')   
    
    prob_feat1,all_loss_1=eval_train_twostage(net1,None,all_loss_1,None,None,gmm4loss)
    prob1=prob_feat1
    prob_feat2,all_loss_2=eval_train_twostage(net2,None,all_loss_2,None,None,gmm4loss)          
    prob2=prob_feat2
    pred1 = (prob1 > args.p_threshold)      
    pred2 = (prob2 > args.p_threshold)      


    stats_log.write(f'selected {pred1.sum()/pred1.shape[0]} \n')
    stats_log.write(f'selected {pred2.sum()/pred2.shape[0]} \n')
    stats_log.flush()

    print('Train Net')
    labeled_trainloader = loader.run('train',pred1,prob1) # co-divide
    train(epoch,net2,net1,optimizer2,labeled_trainloader,num_iter=args.epochs) # train net2         

    print('\nTrain Net2')
    labeled_trainloader = loader.run('train',pred2,prob2) # co-divide
    train(epoch,net1,net2,optimizer1,labeled_trainloader,num_iter=args.epochs) # train net1  


    acc=test(epoch,net1,net2)
    if acc>best_acc:
        best_acc=acc
        save_model(f"{args.save_name}_{args.dataset}_{args.r}_{args.noise_mode}_best","./checkpoint/",net1,net2,optimizer1,optimizer2,epoch)
    
    save_model(f"{args.save_name}_{args.dataset}_{args.r}_{args.noise_mode}_latest","./checkpoint/",net1,net2,optimizer1,optimizer2,epoch)
