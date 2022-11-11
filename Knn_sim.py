import numpy as np
import torch
import torch.nn as nn

class Knn_sim(nn.Module):
    def __init__(self,anchor_feature, anchor_label,graph=50, temperature=0.07, mode='knn',
                 base_temperature=0.07):
        super(Knn_sim, self).__init__()
        self.temperature = temperature
        self.mode = mode
        self.base_temperature = base_temperature
        self.anchor_feature=anchor_feature
        self.anchor_label=anchor_label.contiguous().view(-1, 1)
        self.graph=graph

    def forward(self, features, labels=None, mask=None,reduction=True):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        labels = labels.contiguous().view(-1, 1)

        contrast_feature = features #torch.cat(torch.unbind(features, dim=1), dim=0)


        anchor_dot_contrast = torch.matmul(contrast_feature, self.anchor_feature.T)
        if self.mode=='knn':
            anchor_dot_contrast,anchor_dot_contrast_index=torch.topk(anchor_dot_contrast,self.graph,dim=1,)
            mask = torch.eq(labels,self.anchor_label.T).float().to(device)
            mask = torch.gather(mask, 1, anchor_dot_contrast_index)
            mean_log_prob_pos = (mask).sum(1) / self.graph
        elif self.contrast_mode=='knn_sim':
            anchor_dot_contrast,anchor_dot_contrast_index=torch.topk(anchor_dot_contrast,self.graph,dim=1,)
            mask = torch.eq(labels,self.anchor_label.T).float().to(device)
            mask = torch.gather(mask, 1, anchor_dot_contrast_index)
            mean_log_prob_pos = (mask).sum(1) / self.graph

            mean_sim = (anchor_dot_contrast).sum(1) / self.graph
            mean_sim = mean_sim.view(-1)

            mean_log_prob_pos[mean_log_prob_pos != mean_log_prob_pos] = 0
            loss = - mean_log_prob_pos

            if reduction:
                loss = loss.mean()
            else:
                loss = loss.view(-1)
            return loss,mean_sim

        mean_log_prob_pos[mean_log_prob_pos != mean_log_prob_pos] = 0
        loss = - mean_log_prob_pos

        if reduction:
            loss = loss.mean()
        else:
            loss = loss.view(-1)
        return loss


if __name__=="__main__":
    pass