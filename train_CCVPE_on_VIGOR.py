import os
import argparse
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import torch
import torch.nn as nn
import numpy as np
import math
from dataloader import VIGORDataset, transform_grd, transform_sat
from models import CVM_VIGOR as CVM
from losses import infoNCELoss
import scipy.io as scio


torch.manual_seed(17)
np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"The device is: {}".format(device)

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type=int, help='batch size', default=8)
parser.add_argument('--known_ori', choices=('True','False'), default='False')
parser.add_argument('-l', '--learning_rate', type=float, help='learning rate', default=1e-4)
parser.add_argument('--weight_infoNCE', type=float, help='weight on infoNCE loss', default=1e4)
parser.add_argument('--keep_ratio', type=float, help='keep top T% consistent predictions', default=0.8)

args = vars(parser.parse_args())


batch_size = args['batch_size']
known_ori = args['known_ori'] == 'True'
learning_rate = args['learning_rate']
weight_infoNCE = args['weight_infoNCE']
keep_ratio = args['keep_ratio']


pretrained_model = './models/CCVPE/teacher/model.pt'
save_model_path = './models/CCVPE/final_student/'
dataset_root='/home/ziminxia/Work/datasets/VIGOR'
label = 'CCVPE_VIGOR_final_student'
if not os.path.exists('./results'):
    os.makedirs('./results')

vigor = VIGORDataset(root=dataset_root, transform=(transform_grd, transform_sat), known_ori=known_ori)

with open('shuffled_crossarea_index_list.npy', 'rb') as f:
    index_list = np.load(f)
    
with open('predefined_random_rot.npy', 'rb') as f:
    predefined_random_rot = np.load(f)

train_indices = index_list[0: int(len(index_list)*0.7)]
val_indices = index_list[int(len(index_list)*0.7):int(len(index_list)*0.8)]


mat_teacher = scio.loadmat('teacher_prediction_on_trainingset.mat')
mat_auxiliary_student = scio.loadmat('auxiliary_student_prediction_on_trainingset.mat')

pred_row_offsets_teacher = mat_teacher['pred_row_offsets'][0]
pred_col_offsets_teacher = mat_teacher['pred_col_offsets'][0]
pred_row_offsets_auxiliary_student = mat_auxiliary_student['pred_row_offsets'][0]
pred_col_offsets_auxiliary_student = mat_auxiliary_student['pred_col_offsets'][0]

change_in_prediction = np.sqrt((pred_row_offsets_teacher-pred_row_offsets_auxiliary_student)**2 + (pred_col_offsets_teacher-pred_col_offsets_auxiliary_student)**2)
threshold = np.quantile(change_in_prediction, keep_ratio)

idx_kept = []
for i in range(len(train_indices)):
    if change_in_prediction[i] < threshold:
        idx_kept.append(train_indices[i])
        vigor.pred_delta[train_indices[i], 0, 0] = pred_row_offsets_teacher[i]
        vigor.pred_delta[train_indices[i], 0, 1] = pred_col_offsets_teacher[i]
 

CVM_model = CVM(device)
CVM_model.load_state_dict(torch.load(pretrained_model))
CVM_model.to(device)
for param in CVM_model.parameters():
    param.requires_grad = True
    

torch.cuda.empty_cache()

params = [p for p in CVM_model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=learning_rate, betas=(0.9, 0.999))

global_step = 0
# with torch.autograd.set_detect_anomaly(True):

for epoch in range(3):  # loop over the dataset multiple times        
    
    # training 
    vigor.predefined_random_rot = None
    vigor.load_gt = False
    vigor_train = Subset(vigor, idx_kept)
    train_dataloader = DataLoader(vigor_train, batch_size=batch_size, shuffle=True)
    
    running_loss = 0.0
    CVM_model.train()
    for i, data in enumerate(train_dataloader, 0):
        grd, sat, gt, gt_with_ori, gt_orientation, city = data
        grd = grd.to(device)
        sat = sat.to(device)
        gt = gt.to(device)
        gt_with_ori = gt_with_ori.to(device)
        gt_orientation = gt_orientation.to(device)

        gt_flattened = torch.flatten(gt, start_dim=1)
        gt_flattened = gt_flattened / torch.sum(gt_flattened, dim=1, keepdim=True)

        gt_bottleneck = nn.MaxPool2d(64, stride=64)(gt_with_ori)
        gt_bottleneck2 = nn.MaxPool2d(32, stride=32)(gt_with_ori)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        logits_flattened, heatmap, ori, matching_score_stacked, matching_score_stacked2, matching_score_stacked3, \
                matching_score_stacked4, matching_score_stacked5, matching_score_stacked6 = CVM_model(grd, sat)        

        loss_infoNCE = infoNCELoss(torch.flatten(matching_score_stacked, start_dim=1), torch.flatten(gt_bottleneck, start_dim=1))
        loss_infoNCE2 = infoNCELoss(torch.flatten(matching_score_stacked2, start_dim=1), torch.flatten(gt_bottleneck2, start_dim=1))

        loss = weight_infoNCE*(loss_infoNCE+loss_infoNCE2)/2    
    
        loss.backward()
        optimizer.step()

        global_step += 1
        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 200 mini-batches
            print(f'[{epoch}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
            running_loss = 0.0
        
    model_dir = save_model_path + str(epoch) + '/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(CVM_model.cpu().state_dict(), model_dir+'model.pt') # saving model
    CVM_model.cuda() # moving model back to GPU 
       
    # evaluation
    vigor.predefined_random_rot = predefined_random_rot
    vigor.load_gt = True
    val_set = Subset(vigor, val_indices)
    val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    CVM_model.eval()

    distance_in_meters = []
    for i, data in enumerate(val_dataloader, 0):
        grd, sat, gt, gt_with_ori, gt_orientation, city = data
        grd = grd.to(device)
        sat = sat.to(device)
        
        with torch.no_grad():
            _, heatmap, ori, _, _, _, _, _, _ = CVM_model(grd, sat)

            gt = gt.detach().numpy() 
            gt_orientation = gt_orientation.detach().numpy() 
            heatmap = heatmap.cpu().detach().numpy()
            ori = ori.cpu().detach().numpy() 
            for batch_idx in range(gt.shape[0]):
                if city[batch_idx] == 'None':
                    pass
                else:
                    current_gt = gt[batch_idx, :, :, :]
                    loc_gt = np.unravel_index(current_gt.argmax(), current_gt.shape)
                    current_pred = heatmap[batch_idx, :, :, :]
                    loc_pred = np.unravel_index(current_pred.argmax(), current_pred.shape)                

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

    
    mean_distance_error = np.mean(distance_in_meters)
    print('epoch: ', epoch, 'Mean distance error on validation set: ', mean_distance_error)
    file = './results/'+label+'_mean_distance_error.txt'
    with open(file,'ab') as f:
        np.savetxt(f, [mean_distance_error], fmt='%4f', header='Validation_set_mean_distance_error_in_meters:', comments=str(epoch)+'_')
    
    median_distance_error = np.median(distance_in_meters)
    print('epoch: ', epoch, 'Median distance error on validation set: ', median_distance_error)
    file = './results/'+label+'_median_distance_error.txt'
    with open(file,'ab') as f:
        np.savetxt(f, [median_distance_error], fmt='%4f', header='Validation_set_median_distance_error_in_meters:', comments=str(epoch)+'_')
    

print('Finished Training')