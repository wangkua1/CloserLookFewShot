import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob

import configs
import backbone
from data.datamgr import AttrDataManager, SimpleDataManager, SetDataManager, FFSDataManager
from methods.baselinetrain import BaselineTrain, BaselineTrainTest
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
from io_utils import model_dict, parse_args, get_resume_file  
import itertools
import json
import torchvision.utils as vutils
import matplotlib.pylab as plt
from tqdm import tqdm

def train(base_loader, eval_loaders_dic, model, optimization, start_epoch, stop_epoch, params):    
    if optimization == 'Adam':
        parameters = list(model.parameters())
        if len(parameters) == 0:
            optimizer = None
        else:
            optimizer = torch.optim.Adam(parameters,lr=params.lr)
    else:
       raise ValueError('Unknown optimization, please define by yourself')

    max_acc = 0       

    for epoch in range(start_epoch,stop_epoch):

        model.eval()
        acc_dict = {'epoch':epoch}
        for k, val_loader in eval_loaders_dic.items():
            if params.db and k.startswith('FFS'):
                D = os.path.join(params.checkpoint_dir, k)
                if not os.path.isdir(D):
                    os.makedirs(D)
                for i, (x, y) in tqdm(enumerate(val_loader), desc='plotting'):
                    if i >= 10:
                        break
                    def _plot(ims, title, name):
                        c, h, w = ims.shape[-3:]
                        grid = vutils.make_grid(ims.reshape(-1, c,h,w), nrow=ims.shape[1], padding=2, normalize=True) 
                        fig,  axs = plt.subplots(1,1,figsize=(50,2))
                        plt.imshow(np.transpose(grid.numpy(), (1,2,0)))
                        plt.tight_layout()
                        plt.grid()
                        plt.xticks([])
                        plt.yticks([])
                        plt.title(title,fontsize=20)
                        plt.savefig(name, bbox_inches='tight', pad_inches=0, format='jpeg')
                        plt.close(fig)
                    x = x[0]
                    fname = os.path.join(D,f'{epoch}_{i}.jpeg')
                    _plot(x, f'{y[0][0]}', fname)


            acc = model.test_loop( val_loader)
            print(f"{k} ACC: {acc}")
            acc_dict[k] = acc
        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)
            
        print('Logging to ======>')
        print(params.checkpoint_dir)

        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)
        # Logging
        # - flush first
        if epoch == 0:
            open(os.path.join(params.checkpoint_dir, 'acc.txt'), "w").close()
        with open(os.path.join(params.checkpoint_dir, 'acc.txt'),'a') as f:
            # f.write(f"{epoch}, {acc}, {max_acc}\n")
            f.write(json.dumps(acc_dict))
            f.write('\n')

        if acc > max_acc : #for baseline and baseline++, we don't use validation here so we let acc = -1
            print("best model! save...")
            max_acc = acc
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

        model.train()
        if optimizer is not None:
            model.train_loop(epoch, base_loader,  optimizer ) #model are called by reference, no need to return 
        



    return model

if __name__=='__main__':
    np.random.seed(10)
    params = parse_args('train')


    if params.dataset == 'cross':
        base_file = configs.data_dir['miniImagenet'] + 'all.json' 
        val_file   = configs.data_dir['CUB'] + 'val.json' 
    elif params.dataset == 'cross_char':
        base_file = configs.data_dir['omniglot'] + 'noLatin.json' 
        val_file   = configs.data_dir['emnist'] + 'val.json' 
    else:
        base_file = configs.data_dir[params.dataset] + 'base.json' 
        val_file   = configs.data_dir[params.dataset] + 'val.json' 
         
    if 'Conv' in params.model:
        if params.dataset in ['omniglot', 'cross_char']:
            image_size = 28
        else:
            image_size = 84
    else:
        image_size = 224

    if params.dataset in ['omniglot', 'cross_char']:
        assert params.model == 'Conv4' and not params.train_aug ,'omniglot only support Conv4 without augmentation'
        params.model = 'Conv4S'

    optimization = 'Adam'

    if params.stop_epoch == -1: 
        if params.method in ['baseline', 'baseline++'] :
            if params.dataset in ['omniglot', 'cross_char']:
                params.stop_epoch = 5
            elif params.dataset in ['CUB']:
                params.stop_epoch = 200 # This is different as stated in the open-review paper. However, using 400 epoch in baseline actually lead to over-fitting
            elif params.dataset in ['miniImagenet', 'cross']:
                params.stop_epoch = 400
            else:
                params.stop_epoch = 400 #default
        else: #meta-learning methods
            if params.n_shot == 1:
                params.stop_epoch = 600
            elif params.n_shot == 5:
                params.stop_epoch = 400
            else:
                params.stop_epoch = 600 #default

    eval_loaders_dic = {}
    # FFS
    test_ffs_params     = dict(n_way = 1, n_support = params.n_shot, n_episodes = 20)
    # for attr_split, example_split in tqdm(itertools.product(['train', 'val', 'test'],['base']), desc='Loading Val Loaders'):
    for attr_split, example_split in itertools.product(['train', 'val', 'test'],['base','val','novel']):
        val_loader = FFSDataManager(params.x_type, image_size, attr_split=attr_split, attr_split_file=params.attr_split_file, n_query = 15, **test_ffs_params).get_data_loader( configs.data_dir[params.dataset] + f'{example_split}.json' , aug = False) 
        eval_loaders_dic[f"FFS,attr={attr_split},example={example_split}"] = val_loader
        
    # FS
    # test_few_shot_params     = dict(n_way = params.test_n_way, n_support = params.n_shot, n_episodes = 20) 
    # for  example_split in ['base','val','novel']:
    #     val_loader = SetDataManager(image_size, n_query = 15, **test_few_shot_params).get_data_loader( configs.data_dir[params.dataset] + f'{example_split}.json' , aug = False) 
    #     eval_loaders_dic[f"FS,example={example_split}"] = val_loader
     

    if params.method in ['baseline', 'baseline++'] :
        base_datamgr    = SimpleDataManager(image_size, batch_size = 16)
        base_loader     = base_datamgr.get_data_loader( base_file , aug = params.train_aug )

        model  = BaselineTrainTest( model_dict[params.model], params.num_classes, params.test_n_way, params.n_shot, loss_type = 'softmax' if params.method == 'baseline' else 'dist')

    elif params.method == 'attr':
        base_datamgr    = AttrDataManager(image_size, params.train_attr_split, attr_split_file=params.attr_split_file, batch_size = 16)
        base_loader     = base_datamgr.get_data_loader( base_file , aug = params.train_aug )
        

        model  = BaselineTrainTest( model_dict[params.model],params.num_classes,  params.test_n_way, params.n_shot, loss_type= 'bce')

    elif params.method in ['protonet','matchingnet','relationnet', 'relationnet_softmax', 'maml', 'maml_approx']:
        n_query = max(1, int(16* params.test_n_way/params.train_n_way)) #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
        
        if params.train_ffs:
            train_ffs_params    = dict(n_way = 1, n_support = params.n_shot) 
            base_datamgr            = FFSDataManager(params.x_type, image_size, attr_split='train',attr_split_file=params.attr_split_file,  n_query = 15, **train_ffs_params)
            base_loader             = base_datamgr.get_data_loader( base_file , aug = params.train_aug ) 
            # For initialzing models below
            assert params.train_n_way == 2
            train_few_shot_params    = dict(n_way = params.train_n_way, n_support = params.n_shot) 
        else:
            train_few_shot_params    = dict(n_way = params.train_n_way, n_support = params.n_shot) 
            base_datamgr            = SetDataManager(image_size, n_query = n_query,  **train_few_shot_params)
            base_loader             = base_datamgr.get_data_loader( base_file , aug = params.train_aug )
        
        if params.method == 'protonet':
            model           = ProtoNet( model_dict[params.model], **train_few_shot_params )
        elif params.method == 'matchingnet':
            model           = MatchingNet( model_dict[params.model], **train_few_shot_params )
        elif params.method in ['relationnet', 'relationnet_softmax']:
            if params.model == 'Conv4': 
                feature_model = backbone.Conv4NP
            elif params.model == 'Conv6': 
                feature_model = backbone.Conv6NP
            elif params.model == 'Conv4S': 
                feature_model = backbone.Conv4SNP
            else:
                feature_model = lambda: model_dict[params.model]( flatten = False )
            loss_type = 'mse' if params.method == 'relationnet' else 'softmax'

            model           = RelationNet( feature_model, loss_type = loss_type , **train_few_shot_params )
        elif params.method in ['maml' , 'maml_approx']:
            backbone.ConvBlock.maml = True
            backbone.SimpleBlock.maml = True
            backbone.BottleneckBlock.maml = True
            backbone.ResNet.maml = True
            model           = MAML(  model_dict[params.model], approx = (params.method == 'maml_approx') , **train_few_shot_params )
            if params.dataset in ['omniglot', 'cross_char']: #maml use different parameter in omniglot
                model.n_task     = 32
                model.task_update_num = 1
                model.train_lr = 0.1
    else:
       raise ValueError('Unknown method')

    model = model.cuda()

    if params.checkpoint_dir == '':
        params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)
        if params.train_aug:
            params.checkpoint_dir += '_aug'
        if not params.method  in ['baseline', 'baseline++']: 
            params.checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch
    if params.method == 'maml' or params.method == 'maml_approx' :
        stop_epoch = params.stop_epoch * model.n_task #maml use multiple tasks in one update 

    if params.resume:
        resume_file = get_resume_file(params.checkpoint_dir)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch']+1
            model.load_state_dict(tmp['state'])
    elif params.warmup: #We also support warmup from pretrained baseline feature, but we never used in our paper
        baseline_checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, 'baseline')
        if params.train_aug:
            baseline_checkpoint_dir += '_aug'
        warmup_resume_file = get_resume_file(baseline_checkpoint_dir)
        tmp = torch.load(warmup_resume_file)
        if tmp is not None: 
            state = tmp['state']
            state_keys = list(state.keys())
            for i, key in enumerate(state_keys):
                if "feature." in key:
                    newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
                    state[newkey] = state.pop(key)
                else:
                    state.pop(key)
            model.feature.load_state_dict(state)
        else:
            raise ValueError('No warm_up file')

    model = train(base_loader, eval_loaders_dic,  model, optimization, start_epoch, stop_epoch, params)
