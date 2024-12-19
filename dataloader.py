import os
from torch.utils.data import Dataset
import PIL.Image
from torchvision import transforms
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import random
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.manual_seed(17)
np.random.seed(0)

transform_grd = transforms.Compose([
    transforms.Resize([320, 640]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
])

transform_sat = transforms.Compose([
    transforms.Resize([512, 512]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
])

class VIGORDataset(Dataset):
    def __init__(self, root='./datasets/VIGOR', label_root = 'splits_new', transform=None, load_gt=True, known_ori=True):
        self.root = root
        self.label_root = label_root
        self.known_ori = known_ori
        if transform != None:
            self.grdimage_transform = transform[0]
            self.satimage_transform = transform[1]
            
        self.city_list = ['SanFrancisco', 'Chicago']
        # load sat list
        self.sat_list = []
        self.sat_index_dict = {}

        idx = 0
        for city in self.city_list:
            sat_list_fname = os.path.join(self.root, label_root, city, 'satellite_list.txt')
            with open(sat_list_fname, 'r') as file:
                for line in file.readlines():
                    self.sat_list.append(os.path.join(self.root, city, 'satellite', line.replace('\n', '')))
                    self.sat_index_dict[line.replace('\n', '')] = idx
                    idx += 1
            print('InputData::__init__: load', sat_list_fname, idx)
        self.sat_list = np.array(self.sat_list)
        self.sat_data_size = len(self.sat_list)
        print('Sat loaded, data size:{}'.format(self.sat_data_size))

        # load grd list  
        self.grd_list = []
        self.label = []
        self.sat_cover_dict = {}
        self.gt_delta = []
        idx = 0
        for city in self.city_list:
            # load grd panorama list
            label_fname = os.path.join(self.root, self.label_root, city, 'pano_label_balanced.txt')
                
            with open(label_fname, 'r') as file:
                for line in file.readlines():
                    data = np.array(line.split(' '))
                    label = []
                    for i in [1, 4, 7, 10]:
                        label.append(self.sat_index_dict[data[i]])
                    label = np.array(label).astype(int)
                    delta = np.array([data[2:4], data[5:7], data[8:10], data[11:13]]).astype(float)
                    self.grd_list.append(os.path.join(self.root, city, 'panorama', data[0]))
                    self.label.append(label)
                    self.gt_delta.append(delta)
                    if not label[0] in self.sat_cover_dict:
                        self.sat_cover_dict[label[0]] = [idx]
                    else:
                        self.sat_cover_dict[label[0]].append(idx)
                    idx += 1
            print('InputData::__init__: load ', label_fname, idx)
        self.data_size = len(self.grd_list)
        print('Grd loaded, data size:{}'.format(self.data_size))
        self.label = np.array(self.label)
        self.gt_delta = np.array(self.gt_delta)
        
        self.pred_delta = np.zeros(np.shape(self.gt_delta))
        self.load_gt = load_gt
        self.predefined_random_rot = None
    
    def __len__(self):
        return self.data_size
    
    def __getitem__(self, idx):        
        # full ground panorama
        try:
            grd = PIL.Image.open(os.path.join(self.grd_list[idx]))
            grd = grd.convert('RGB')
        except:
            print('unreadable image')
            grd = PIL.Image.new('RGB', (320, 640))
        grd = self.grdimage_transform(grd)
        # generate a random rotation 
        if self.known_ori:
            rotation = 0
        else:
            if self.predefined_random_rot is None:
                rotation = np.random.uniform(low=0.0, high=1.0)
            else:
                rotation = self.predefined_random_rot[idx]

        grd = torch.roll(grd, (torch.round(torch.as_tensor(rotation)*grd.size()[2]).int()).item(), dims=2)
                
        orientation_angle = rotation * 360 # 0 means heading North, counter-clockwise increasing
        if orientation_angle < 0:
            orientation_angle += 360
        # satellite
        pos_index = 0
        sat = PIL.Image.open(os.path.join(self.sat_list[self.label[idx][pos_index]]))
        if self.load_gt == True:
            [row_offset, col_offset] = self.gt_delta[idx, pos_index] # delta = [delta_lat, delta_lon]
        else:
            [row_offset, col_offset] = self.pred_delta[idx, pos_index] # delta = [delta_lat, delta_lon]
                
                
        sat = sat.convert('RGB')
        width_raw, height_raw = sat.size
        
        sat = self.satimage_transform(sat)
        _, height, width = sat.size()
        
        row_offset = np.round(row_offset/height_raw*height)
        col_offset = np.round(col_offset/width_raw*width)
        
        # groundtruth location on the satellite map        
        # Gaussian GT        
        gt = np.zeros([1, height, width], dtype=np.float32)
        gt_with_ori = np.zeros([20, height, width], dtype=np.float32)
        x, y = np.meshgrid(np.linspace(-width/2+col_offset,width/2+col_offset,width), np.linspace(-height/2-row_offset,height/2-row_offset,height))
        d = np.sqrt(x*x+y*y)
        sigma, mu = 4, 0.0
        gt[0, :, :] = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
        gt = torch.tensor(gt)
        
        index = int(orientation_angle // 18)
        ratio = (orientation_angle % 18) / 18
        if index == 0:
            gt_with_ori[0, :, :] = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) ) * (1-ratio)
            gt_with_ori[19, :, :] = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) ) * ratio
        else:
            gt_with_ori[20-index, :, :] = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) ) * (1-ratio)
            gt_with_ori[20-index-1, :, :] = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) ) * ratio
        gt_with_ori = torch.tensor(gt_with_ori)
        
        orientation = torch.full([2, height, width], np.cos(orientation_angle * np.pi/180))
        orientation[1,:,:] = np.sin(orientation_angle * np.pi/180)
        
        if 'NewYork' in self.grd_list[idx]:
            city = 'NewYork'
        elif 'Seattle' in self.grd_list[idx]:
            city = 'Seattle'
        elif 'SanFrancisco' in self.grd_list[idx]:
            city = 'SanFrancisco'
        elif 'Chicago' in self.grd_list[idx]:
            city = 'Chicago'
            
        return grd, sat, gt, gt_with_ori, orientation, city
