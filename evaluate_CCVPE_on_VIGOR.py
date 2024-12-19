import os
import argparse
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import torch
import torch.nn as nn
import numpy as np
import math
from dataloader import VIGORDataset, transform_grd, transform_sat
from models import CVM_VIGOR, CVM_VIGOR_ori_prior
from losses import infoNCELoss, cross_entropy_loss
import scipy.io as scio


torch.manual_seed(17)
np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"The device is: {}".format(device)

parser = argparse.ArgumentParser()
parser.add_argument('--inference_on', choices=('train', 'val', 'test'), default='test')
parser.add_argument('-b', '--batch_size', type=int, help='batch size', default=16)
parser.add_argument('--known_ori', choices=('True','False'), default='True')
parser.add_argument('--model', choices=('teacher', 'auxiliary_student', 'final_student'), default=None)



args = vars(parser.parse_args())

if args['inference_on']=='train': 
    label = args['model'] + '_prediction_on_trainingset'


batch_size = args['batch_size']
inference_on = args['inference_on']
known_ori = args['known_ori'] == 'True'
selected_model = args['model']

model_path = '/home/ziminxia/Work/experiments/Adapting_CVL/models/CCVPE/'+selected_model+'/model.pt'
# model_path = '/home/ziminxia/Work/experiments/Weakly_supervised_learning/main_experiment_results/Gaussian_sig4_lr_0.0001_from_pretrainedmodel_infoNCE12/1/model.pt'

# model_path = '/home/ziminxia/Work/experiments/Adapting_CVL/models/CCVPE/final_student/1/model.pt'
dataset_root = '/home/ziminxia/Work/datasets/VIGOR'


vigor = VIGORDataset(root=dataset_root, transform=(transform_grd, transform_sat), known_ori=known_ori)

with open('shuffled_crossarea_index_list.npy', 'rb') as f:
    index_list = np.load(f)
    
with open('predefined_random_rot.npy', 'rb') as f:
    predefined_random_rot = np.load(f)

train_indices = index_list[0: int(len(index_list)*0.7)]
val_indices = index_list[int(len(index_list)*0.7):int(len(index_list)*0.8)]
test_indices = index_list[int(len(index_list)*0.8):]
train_set = Subset(vigor, train_indices)
val_set = Subset(vigor, val_indices)
test_set = Subset(vigor, test_indices)

if inference_on == 'train':
    dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
elif inference_on == 'val':
    dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
elif inference_on == 'test':
    dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

vigor.predefined_random_rot = predefined_random_rot


torch.cuda.empty_cache()

if known_ori:
    CVM_model = CVM_VIGOR_ori_prior(device)
else:
    CVM_model = CVM_VIGOR(device)
CVM_model.load_state_dict(torch.load(model_path))
CVM_model.to(device)
CVM_model.eval()

distance_in_meters = []
pred_row_offsets = []
pred_col_offsets = []

for i, data in enumerate(dataloader, 0):
    with torch.no_grad():
        print(i)
        grd, sat, gt, gt_with_ori, gt_orientation, city = data
        grd = grd.to(device)
        sat = sat.to(device)

        gt_with_ori = gt_with_ori.to(device)

        gt_flattened = torch.flatten(gt, start_dim=1)
        gt_flattened = gt_flattened / torch.sum(gt_flattened, dim=1, keepdim=True)

        logits_flattened, heatmap, ori, matching_score_stacked, matching_score_stacked2, matching_score_stacked3, matching_score_stacked4, matching_score_stacked5, matching_score_stacked6 = CVM_model(grd, sat)

        gt = gt.cpu().detach().numpy() 
        gt_with_ori = gt_with_ori.cpu().detach().numpy() 
        gt_orientation = gt_orientation.cpu().detach().numpy() 
        heatmap = heatmap.cpu().detach().numpy()
        for batch_idx in range(gt.shape[0]):
            if city[batch_idx] == 'None':
                pass
            else:
                current_gt = gt[batch_idx, :, :, :]
                loc_gt = np.unravel_index(current_gt.argmax(), current_gt.shape)
                current_pred = heatmap[batch_idx, :, :, :]

                loc_pred = np.unravel_index(current_pred.argmax(), current_pred.shape)

                pred_row_offsets.append((loc_pred[1] - 256) / 512 * 640)
                pred_col_offsets.append((256 - loc_pred[2]) / 512 * 640)

                pixel_distance = np.sqrt((loc_gt[1]-loc_pred[1])**2+(loc_gt[2]-loc_pred[2])**2)
                if city[batch_idx] == 'NewYork':
                    meter_distance = pixel_distance * 0.113248 / 512 * 640
                elif city[batch_idx] == 'Seattle':
                     meter_distance = pixel_distance * 0.100817 / 512 * 640
                elif city[batch_idx] == 'SanFrancisco':
                    meter_distance = pixel_distance * 0.118141 / 512 * 640
                elif city[batch_idx] == 'Chicago':
                    meter_distance = pixel_distance * 0.111262 / 512 * 640
                distance_in_meters.append(meter_distance) 


            
print(np.mean(distance_in_meters))   
print(np.median(distance_in_meters))

if inference_on == 'train':
    scio.savemat(label+'.mat', {'pred_row_offsets': np.array(pred_row_offsets), 'pred_col_offsets': np.array(pred_col_offsets)})
