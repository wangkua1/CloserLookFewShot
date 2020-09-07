# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import json
import numpy as np
import torchvision.transforms as transforms
import os
from collections import defaultdict
import itertools
identity = lambda x:x




class AttrDataset:
    def __init__(self, data_file, attr_split, transform, target_transform=identity):
        assert attr_split in ['train', 'all']

        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        
        base_path = os.path.join(os.path.split(data_file)[0], 'CUB_200_2011')

        # 
        self.attr_mask = np.zeros((312,))
        attr_split_dic = json.load(open(os.path.join(base_path,'attr_splits.json'),'r'))
        if attr_split == 'train':
            considered_attrs = attr_split_dic[attr_split]
        elif attr_split ==  'all':
            considered_attrs = attr_split_dic['train'] + attr_split_dic['val'] + attr_split_dic['test']
        else:
            raise 
        raw_attrs = open(os.path.join(base_path, 'attributes', 'attributes.txt')).readlines()
        for n, r in enumerate(raw_attrs):
            attr, _ = r.strip().split(' ')[1].split('::')
            if attr in considered_attrs:
                self.attr_mask[n] = 1

        image_attribute_labels = open(os.path.join(base_path, 'attributes', 'image_attribute_labels.txt')).readlines()
        image_id2attr_dic = defaultdict(lambda: np.zeros((312,)))
        for i in range(len(image_attribute_labels)):
            image_id, attr_id, val = image_attribute_labels[i].strip().split(' ')[:3]
            image_id2attr_dic[int(image_id)][int(attr_id)-1] = int(val)


        image_id_paths = open(os.path.join(base_path, 'images.txt')).readlines()
        image_path2id_dic = {}
        for i in range(len(image_id_paths)):
            id, path = image_id_paths[i].strip().split(' ')
            image_path2id_dic[path] = int(id)

        self.image_id2attr_dic = image_id2attr_dic
        self.image_path2id_dic = image_path2id_dic

        self.transform = transform
        self.target_transform = target_transform

    def _path2attr(self, image_path):
        basepath, name = os.path.split(image_path)
        dirname = os.path.split(basepath)[1]
        path = os.path.join(dirname, name)
        return self.image_id2attr_dic[self.image_path2id_dic[path]]

    def __getitem__(self,i):
        image_path = os.path.join(self.meta['image_names'][i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = torch.from_numpy(self.attr_mask*self._path2attr(image_path)).float()
        return img, target

    def __len__(self):
        return len(self.meta['image_names'])

class SimpleDataset:
    def __init__(self, data_file, transform, target_transform=identity):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self,i):
        image_path = os.path.join(self.meta['image_names'][i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.meta['image_labels'][i])
        return img, target

    def __len__(self):
        return len(self.meta['image_names'])

class FFSDataset:
    def __init__(self, data_file, attr_split, batch_size, transform, n_episodes):
        self.n_episodes = n_episodes
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        
        


        base_path = os.path.join(os.path.split(data_file)[0], 'CUB_200_2011')

        #
        image_attribute_labels = open(os.path.join(base_path, 'attributes', 'image_attribute_labels.txt')).readlines()
        image_attribute_matrix = np.zeros((11788, 312))
        for i in range(len(image_attribute_labels)):
            image_id, attr_id, val = image_attribute_labels[i].strip().split(' ')[:3]
            image_attribute_matrix[int(image_id)-1, int(attr_id)-1] = int(val)

        # 
        attr_val_id_dic = defaultdict(list)
        raw_attrs = open(os.path.join(base_path, 'attributes', 'attributes.txt')).readlines()
        for n, r in enumerate(raw_attrs):
            k,v = r.strip().split(' ')[1].split('::')
            attr_val_id_dic[k].append((v, n+1))

        attr_split_dic = json.load(open(os.path.join(base_path,'attr_splits.json'),'r'))
        considered_attrs = attr_split_dic[attr_split]

        #
        image_id_paths = open(os.path.join(base_path, 'images.txt')).readlines()
        image_id2path_dic = {}
        image_path2id_dic = {}
        for i in range(len(image_id_paths)):
            id, path = image_id_paths[i].strip().split(' ')
            image_id2path_dic[int(id)] = path
            image_path2id_dic[path] = int(id)

        # Collect all image ids for this split
        split_ids = []
        for image_path in self.meta['image_names']:
            tmp, B = os.path.split(image_path)
            _, A = os.path.split(tmp) 
            path = os.path.join(A, B)
            split_ids.append(image_path2id_dic[path])

        #
        example_split = os.path.split(data_file)[1].split('.')[0]
        all_ind_combi_str_to_example_ids_dic = {}
        for attr1, attr2 in itertools.combinations(considered_attrs, 2):
            attr1_inds = [i[1]-1 for i in attr_val_id_dic[attr1]]
            attr2_inds = [i[1]-1 for i in attr_val_id_dic[attr2]]
            ind_combi_strs = [f"{x},{y}" for x, y in list(itertools.product(attr1_inds, attr2_inds))]
            for ind_combi_str in ind_combi_strs:
                # Select all examples
                x,y = [int(i) for i in ind_combi_str.split(',')]
                ids = 1+np.nonzero((image_attribute_matrix[:,x] == 1).astype('int32') * (image_attribute_matrix[:,y] == 1).astype('int32'))[0]
                # Filter for only ids in this split
                ids = list(set(ids).intersection(set(split_ids)))
                if len(ids) >= batch_size:
                    all_ind_combi_str_to_example_ids_dic[ind_combi_str] = [
                        os.path.join('filelists/CUB/CUB_200_2011/images/',image_id2path_dic[id]) for id in ids]
        #
        all_img_paths = []
        for i in range(len(image_id_paths)):
            id, path = image_id_paths[i].strip().split(' ')
            all_img_paths.append(os.path.join('filelists/CUB/CUB_200_2011/images/',path))

        self.sub_meta = all_ind_combi_str_to_example_ids_dic
        self.cl_list = list(self.sub_meta.keys())

        self.sub_dataloader = [] 
        sub_data_loader_params = dict(batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 0, #use main thread only or may receive multiple batches
                                  pin_memory = False)        
        for cl in self.cl_list:
            neg = list(set(all_img_paths).difference(set(self.sub_meta[cl])))
            sub_dataset = PairDataset(self.sub_meta[cl],neg, cl, transform = transform )
            self.sub_dataloader.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )

    def __getitem__(self,i):
        pos, neg, y  = next(iter(self.sub_dataloader[i]))
        x = torch.stack([pos, neg])
        return x, y

    def __len__(self):
        return self.n_episodes

class PairDataset:
    def __init__(self, pos, neg, cl, transform=transforms.ToTensor(), target_transform=identity):
        self.pos = pos
        self.neg = neg
        self.cl = cl 
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self,i):
        #print( '%d -%d' %(self.cl,i))
        pos_img = self.transform(Image.open(os.path.join( self.pos[i])).convert('RGB'))
        neg_img = self.transform(Image.open(os.path.join( self.neg[i])).convert('RGB'))
        target = self.target_transform(self.cl)
        return pos_img, neg_img, target

    def __len__(self):
        return min(len(self.pos), len(self.neg))

class SetDataset:
    def __init__(self, data_file, batch_size, transform):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
 
        self.cl_list = np.unique(self.meta['image_labels']).tolist()

        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        for x,y in zip(self.meta['image_names'],self.meta['image_labels']):
            self.sub_meta[y].append(x)

        self.sub_dataloader = [] 
        sub_data_loader_params = dict(batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 0, #use main thread only or may receive multiple batches
                                  pin_memory = False)        
        for cl in self.cl_list:
            sub_dataset = SubDataset(self.sub_meta[cl], cl, transform = transform )
            self.sub_dataloader.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )

    def __getitem__(self,i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.cl_list)



class SubDataset:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity):
        self.sub_meta = sub_meta
        self.cl = cl 
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self,i):
        #print( '%d -%d' %(self.cl,i))
        image_path = os.path.join( self.sub_meta[i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.cl)
        return img, target

    def __len__(self):
        return len(self.sub_meta)

class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]
