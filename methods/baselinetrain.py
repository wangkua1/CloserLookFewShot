import backbone
import utils

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate

class BaselineTrain(nn.Module):
    def __init__(self, model_func, num_class, loss_type = 'softmax'):
        super(BaselineTrain, self).__init__()
        self.feature    = model_func()
        if loss_type == 'softmax':
            self.classifier = nn.Linear(self.feature.final_feat_dim, num_class)
            self.classifier.bias.data.fill_(0)
            self.loss_fn = nn.CrossEntropyLoss()
        elif loss_type == 'dist': #Baseline ++
            self.classifier = backbone.distLinear(self.feature.final_feat_dim, num_class)
            self.loss_fn = nn.CrossEntropyLoss()
        elif loss_type == 'bce':
            linear = nn.Linear(self.feature.final_feat_dim, 312)
            linear.bias.data.fill_(0)
            self.classifier = nn.Sequential(linear,
                                    nn.Sigmoid())
            self.loss_fn = nn.BCELoss()

        self.loss_type = loss_type 
        self.num_class = num_class
        

    def forward(self,x):
        x    = Variable(x.cuda())
        out  = self.feature.forward(x)
        scores  = self.classifier.forward(out)
        return scores

    def forward_loss(self, x, y):
        scores = self.forward(x)
        y = Variable(y.cuda())
        return self.loss_fn(scores, y )
    
    def train_loop(self, epoch, train_loader, optimizer):
        print_freq = 10
        avg_loss=0

        for i, (x,y) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = self.forward_loss(x, y)
            loss.backward()
            optimizer.step()

            avg_loss = avg_loss+loss.item()

            if i % print_freq==0:
                #print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1)  ))
                     
    def test_loop(self, val_loader):
        return -1 #no validation, just save model during iteration


class BaselineTrainTest(MetaTemplate):
    def __init__(self, model_func,num_class, n_way, n_support, loss_type = 'softmax'):
        super(BaselineTrainTest, self).__init__(model_func,  n_way, n_support)
        self.feature    = model_func()
        if loss_type == 'softmax':
            self.classifier = nn.Linear(self.feature.final_feat_dim, num_class)
            self.classifier.bias.data.fill_(0)
            self.loss_fn = nn.CrossEntropyLoss()
        elif loss_type == 'dist': #Baseline ++
            self.classifier = backbone.distLinear(self.feature.final_feat_dim, num_class)
            self.loss_fn = nn.CrossEntropyLoss()
        elif loss_type == 'bce':
            linear = nn.Linear(self.feature.final_feat_dim, 312)
            linear.bias.data.fill_(0)
            self.classifier = nn.Sequential(linear,
                                    nn.Sigmoid())
            self.loss_fn = nn.BCELoss()
        self.loss_type = loss_type 

    def forward(self,x):
        x    = Variable(x.cuda())
        out  = self.feature.forward(x)
        scores  = self.classifier.forward(out)
        return scores

    def forward_loss(self, x, y):
        scores = self.forward(x)
        y = Variable(y.cuda())
        return self.loss_fn(scores, y )
    
    def train_loop(self, epoch, train_loader, optimizer):
        print_freq = 10
        avg_loss=0

        for i, (x,y) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = self.forward_loss(x, y)
            loss.backward()
            optimizer.step()

            avg_loss = avg_loss+loss.item()

            if i % print_freq==0:
                #print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1)  ))
                     
    def set_forward(self,x,is_feature = False):
        return self.set_forward_adaptation(x,is_feature); #Baseline always do adaptation
 
    def set_forward_adaptation(self,x,is_feature = True):
        # assert is_feature == True, 'Baseline only support testing with feature'
        z_support, z_query  = self.parse_feature(x,is_feature)

        z_support   = z_support.contiguous().view(self.n_way* self.n_support, -1 ).detach()
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 ).detach()

        y_support = torch.from_numpy(np.repeat(range( self.n_way ), self.n_support ))
        y_support = Variable(y_support.cuda())

        if self.loss_type in ['softmax','bce']:
            linear_clf = nn.Linear(self.feat_dim, self.n_way)
        elif self.loss_type == 'dist':        
            linear_clf = backbone.distLinear(self.feat_dim, self.n_way)
        linear_clf = linear_clf.cuda()

        set_optimizer = torch.optim.SGD(linear_clf.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)

        loss_function = nn.CrossEntropyLoss()
        loss_function = loss_function.cuda()
        
        batch_size = 4
        support_size = self.n_way* self.n_support
        for epoch in range(100):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size , batch_size):
                set_optimizer.zero_grad()
                selected_id = torch.from_numpy( rand_id[i: min(i+batch_size, support_size) ]).cuda()
                z_batch = z_support[selected_id]
                y_batch = y_support[selected_id] 
                scores = linear_clf(z_batch)
                loss = loss_function(scores,y_batch)
                loss.backward()
                set_optimizer.step()
        scores = linear_clf(z_query)
        return scores


    def set_forward_loss(self,x):
        raise ValueError('Baseline predict on pretrained feature and do not support finetune backbone')

