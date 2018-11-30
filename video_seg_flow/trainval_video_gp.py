import os
import time
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn
from dataloader.video_generator import DataGenerator
from dataloader.imagereader import DAVIS, YOUTUBE, versa
from utils.lr_scheduler import LRScheduler
from utils import helper

#================ load your net here ===================
# from net.videoseg import Videoseg_diff as Videoseg
from net.lw_mobilenet_gp import mbv2 as network
# from net.lw_resnet import rf_lw50 as network
# from net.lw_resnet import rf_lw101 as network
# from net.lw_resnet import rf_lw152 as network

def get_warp_label(flow1, flow2, label1): #H,W,C  h*w*2

    flow_shape = flow1.shape
    label_shape = label1.shape  
    height, width = flow1.shape[0], flow1.shape[1]

    label2 = torch.zeros(label_shape).cuda(async=True)   #label2 = np.zeros_like(label1, dtype=label1.dtype)
    flow_t = torch.zeros(flow_shape).cuda(async=True)    #flow_t = np.zeros_like(flow1, dtype=flow1.dtype)

    #grid = np.indices((height, width)).swapaxes(0, 1).swapaxes(1, 2)
    h_grid = torch.arange(0, height).cuda(async=True)
    w_grid = torch.arange(0, width).cuda(async=True)
    h_grid = h_grid.repeat(width, 1).permute(1,0) #.unsqueeze(0)
    w_grid = w_grid.repeat(height,1)              #.unsqueeze(0)
    grid = torch.stack((h_grid,w_grid),0).permute(1,2,0) #float3
    #grid = torch.cat([h_grid, w_grid],0).permute(1,2,0)

    dx = grid[:, :, 0] + flow2[:, :, 1].long()
    dy = grid[:, :, 1] + flow2[:, :, 0].long()
    sx = torch.floor(dx.float()).cuda(async=True) #float32 #sx = np.floor(dx).astype(int)
    sy = torch.floor(dy.float()).cuda(async=True) 

    valid = (sx >= 0) & (sx < height - 1) & (sy >= 0) & (sy < width - 1) #H* W 512 x 512 uint8

    # sx_mat = np.dstack((sx, sx + 1, sx, sx + 1)).clip(0, height - 1)
    # sy_mat = np.dstack((sy, sy, sy + 1, sy + 1)).clip(0, width - 1)
    # sxsy_mat = np.abs((1 - np.abs(sx_mat - dx[:, :, np.newaxis])) *
    #                   (1 - np.abs(sy_mat - dy[:, :, np.newaxis])))

    sx_mat = torch.stack((sx, sx + 1, sx, sx + 1),dim=2).clamp(0, height - 1).cuda(async=True)  #torch.float32
    sy_mat = torch.stack((sy, sy, sy + 1, sy + 1),dim=2).clamp(0, width - 1).cuda(async=True)
    sxsy_mat = torch.abs((1 - torch.abs(sx_mat - dx.float().unsqueeze(0).permute(1,2,0))) *
                          (1 - torch.abs(sy_mat - dy.float().unsqueeze(0).permute(1,2,0)))).cuda(async=True)

    for i in range(4):
        flow_t = flow_t.long() + sxsy_mat.long()[:, :, i].unsqueeze(0).permute(1,2,0) * flow1.long()[sx_mat.long()[:, :, i], sy_mat.long()[:, :, i], :]

    valid = valid & (torch.norm(flow_t.float()[:, :, [1, 0]] + torch.stack((dx.float(),dy.float()),dim=2) - grid.float(), dim=2, keepdim=True).squeeze(2) < 100)

    flow_t = (flow2.float() - flow_t.float()) / 2.0
    dx = grid.long()[:, :, 0] + flow_t.long()[:, :, 1]
    dy = grid.long()[:, :, 1] + flow_t.long()[:, :, 0]
    valid = valid & (dx >= 0) & (dx < height - 1) & (dy >= 0) & (dy < width - 1)

    label2[valid, :] = label1.float()[dx[valid].long(), dy[valid].long(), :]
    
    return label2 #HW3  cuda

def print_log(epoch,
              lr,
              train_metrics,
              train_warpmetrics,
              train_featuremetrics,
              train_time,
              val_metrics=None,
              val_warpmetrics=None,
              val_featuremetrics=None,
              val_time=None,
              val_iou=None,
              save_dir=None,
              log_mode=None):
    if epoch > 1:
        log_mode = 'a'
    train_metrics = np.mean(train_metrics, axis=0)
    train_warpmetrics = np.mean(train_warpmetrics, axis=0)
    train_featuremetrics = np.mean(train_featuremetrics, axis=0)
    str0 = 'Epoch %03d (lr %.7f)' % (epoch, lr)
    str0 += f', Train: time {train_time:3.2f} loss: {train_metrics:2.4f} warp_loss: {train_warpmetrics:2.4f} feature_loss: {train_featuremetrics:2.4f}'
    f = open(save_dir + 'train_log.txt', log_mode)
    if val_time is not None:
        val_metrics = np.mean(val_metrics, axis=0)
        val_warpmetrics = np.mean(val_warpmetrics, axis=0)
        val_featuremetrics = np.mean(val_featuremetrics, axis=0)
        str0 += f', Validation: time {val_time:3.2f} loss: {val_metrics:2.4f} warp_loss: {val_warpmetrics:2.4f} feature_loss: {val_featuremetrics:2.4f} iou: {val_iou:.3f}'
    print(str0)
    f.write(str0)
    f.write('\n')
    f.close()



def train(data_loader, net, loss, loss_l1, lamda, beta, optimizer, lr):
    start_time = time.time()
    print("IN Training")
    net.train()
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    metrics = []
    warp_metrics = []
    warp_feature_metrics = []
    for i, batch in enumerate(tqdm(data_loader, ncols=80, ascii=True)):

        prev, cur, heatmaps, prev_label, flow1, flow2 = batch['prev'], batch['cur'], batch['label'], batch['prev_label'], batch['flow1'], batch['flow2']
        batch_size = len(prev)
        prev = prev.cuda(async=True)
        cur = cur.cuda(async=True)
        heatmaps = heatmaps.cuda(async=True)
        prev_label = prev_label.cuda(async=True).float()
        flow1 = flow1.cuda(async=True)
        flow2 = flow2.cuda(async=True)

        with torch.no_grad():
            net.eval()
            prev_prediction, prev_feature = net.forward_all(prev)
            prev_global, _ = net.global_feature(prev, prev_label, prev_feature)
            #prev_prediction, prev_global, prev_feature = net.prediction(prev, None)
            #prev_feature = net.feature(prev)
        pf3,pf4,pf5,pf6,pf7,pf8 = prev_feature

        net.train()        
        cur_prediction, cur_feature = net.forward_all(cur, prev_global)
        cf3,cf4,cf5,cf6,cf7,cf8 = cur_feature
        #H W C N
        warp_prev_output = [get_warp_label(flow1[i].permute(1,2,0), flow2[i].permute(1,2,0), prev_prediction[i].permute(1,2,0))for i in range(batch_size) ] 
        warp_prev_output = torch.stack(warp_prev_output, dim=3).permute(3,2,0,1)  #float32, cuda,  N 2 512 512
        
        wf3 = []
        wf4 = []
        wf5 = []
        wf6 = []
        wf7 = []
        wf8 = []
        
        for i in range(batch_size):
        	f1_3 = F.avg_pool2d(flow1[i], (4,4))/4
        	f1_4 = F.avg_pool2d(flow1[i], (8,8))/8
        	f1_5 = F.avg_pool2d(flow1[i], (16,16))/16
        	f1_7 = F.avg_pool2d(flow1[i], (32,32))/32
        	f2_3 = F.avg_pool2d(flow2[i], (4,4))/4
        	f2_4 = F.avg_pool2d(flow2[i], (8,8))/8
        	f2_5 = F.avg_pool2d(flow2[i], (16,16))/16
        	f2_7 = F.avg_pool2d(flow2[i], (32,32))/32
         
        	wf3.append(get_warp_label(f1_3.permute(1,2,0), f2_3.permute(1,2,0), pf3[i].permute(1,2,0)))
        	wf4.append(get_warp_label(f1_4.permute(1,2,0), f2_4.permute(1,2,0), pf4[i].permute(1,2,0)))
        	wf5.append(get_warp_label(f1_5.permute(1,2,0), f2_5.permute(1,2,0), pf5[i].permute(1,2,0)))
        	wf6.append(get_warp_label(f1_5.permute(1,2,0), f2_5.permute(1,2,0), pf6[i].permute(1,2,0)))
        	wf7.append(get_warp_label(f1_7.permute(1,2,0), f2_7.permute(1,2,0), pf7[i].permute(1,2,0)))
        	wf8.append(get_warp_label(f1_7.permute(1,2,0), f2_7.permute(1,2,0), pf8[i].permute(1,2,0)))
         
        wf3 = torch.stack(wf3, dim=3).permute(3,2,0,1)
        wf4 = torch.stack(wf4, dim=3).permute(3,2,0,1)
        wf5 = torch.stack(wf5, dim=3).permute(3,2,0,1)
        wf6 = torch.stack(wf6, dim=3).permute(3,2,0,1)
        wf7 = torch.stack(wf7, dim=3).permute(3,2,0,1)
        wf8 = torch.stack(wf8, dim=3).permute(3,2,0,1)
        
        '''
        softmax = nn.Softmax().cuda()
        cur_pre_softmax = softmax(cur_prediction)
        warp_res =softmax(warp_prev_output)
        #l1_loss = loss_L1(cur_pre_softmax, warp_res)
        
        wfm3 = torch.argmax(wf3, dim = 1)
        wfm4 = torch.argmax(wf4, dim = 1)
        wfm5 = torch.argmax(wf5, dim = 1)
        wfm6 = torch.argmax(wf6, dim = 1)
        wfm7 = torch.argmax(wf7, dim = 1)
        wfm8 = torch.argmax(wf8, dim = 1)
        '''
        warp_output_max = torch.argmax(warp_prev_output, dim = 1)
        heatmaps = heatmaps[:,0,:,:]
        
        ce_loss = loss(cur_prediction, heatmaps)
        warp_output_loss = loss(cur_prediction, warp_output_max)
        
        wf3_loss = loss_l1(cf3, wf3)
        wf4_loss = loss_l1(cf4, wf4)
        wf5_loss = loss_l1(cf5, wf5)
        wf6_loss = loss_l1(cf6, wf6)
        wf7_loss = loss_l1(cf7, wf7)
        wf8_loss = loss_l1(cf8, wf8) 
        
        wf_loss = wf3_loss+wf4_loss+wf5_loss+wf6_loss+wf7_loss+wf8_loss
        #print("l1 loss: ", l1_loss)
        #print("ce_loss: ", ce_loss)
        loss_output = ce_loss + lamda * warp_output_loss + beta * wf_loss

        optimizer.zero_grad()
        loss_output.backward()
        optimizer.step()
        metrics.append(loss_output.item())
        warp_metrics.append(warp_output_loss.item())
        warp_feature_metrics.append(wf_loss.item())
    end_time = time.time()
    metrics = np.asarray(metrics, np.float32)
    warp_metrics = np.asarray(warp_metrics, np.float32)
    warp_feature_metrics = np.asarray(warp_feature_metrics, np.float32)
    return metrics, warp_metrics, warp_feature_metrics, end_time - start_time

def validate(data_loader, net, loss, loss_l1, lamda, beta, epoch):
    start_time = time.time()
    print("IN VALIDATION")
    net.eval()
    metrics = []
    warp_metrics = []
    warp_feature_metrics = []
    iou = 0
    for i, batch in enumerate(tqdm(data_loader, ncols=80, ascii=True)):
        prev, cur, heatmaps, prev_label, flow1, flow2 = batch['prev'], batch['cur'], batch['label'], batch['prev_label'], batch['flow1'], batch['flow2']
        #print("shape:",prev.shape,heatmaps.shape, flow1.shape)
        #print("dtype:",prev.dtype, cur.dtype, heatmaps.dtype, flow1.dtype, flow2.dtype)
        batch_size = len(prev)
        prev = prev.cuda(async=True)
        cur = cur.cuda(async=True)
        heatmaps = heatmaps.cuda(async=True)
        prev_label = prev_label.cuda(async=True).float()
        flow1 = flow1.cuda(async=True)
        flow2 = flow2.cuda(async=True)
        
        # with torch.no_grad():
        #     pf3,pf4,pf5,pf6,pf7,pf8, prev_prediction = net(prev)
        # cf3,cf4,cf5,cf6,cf7,cf8, cur_prediction = net(cur)

        prev_prediction, prev_feature = net.forward_all(prev)
        prev_global, _ = net.global_feature(prev, prev_label, prev_feature)

        pf3,pf4,pf5,pf6,pf7,pf8 = prev_feature

      
        cur_prediction, cur_feature = net.forward_all(cur, prev_global)
        cf3,cf4,cf5,cf6,cf7,cf8 = cur_feature

        #H W C N
        warp_prev_output = [get_warp_label(flow1[i].permute(1,2,0), flow2[i].permute(1,2,0), prev_prediction[i].permute(1,2,0))for i in range(batch_size) ] 
        warp_prev_output = torch.stack(warp_prev_output, dim=3).permute(3,2,0,1)  #float32, cuda,  N 2 512 512

        wf3 = []
        wf4 = []
        wf5 = []
        wf6 = []
        wf7 = []
        wf8 = []
        
        for i in range(batch_size):
        	f1_3 = F.avg_pool2d(flow1[i], (4,4))/4
        	f1_4 = F.avg_pool2d(flow1[i], (8,8))/8
        	f1_5 = F.avg_pool2d(flow1[i], (16,16))/16
        	f1_7 = F.avg_pool2d(flow1[i], (32,32))/32
        	f2_3 = F.avg_pool2d(flow2[i], (4,4))/4
        	f2_4 = F.avg_pool2d(flow2[i], (8,8))/8
        	f2_5 = F.avg_pool2d(flow2[i], (16,16))/16
        	f2_7 = F.avg_pool2d(flow2[i], (32,32))/32
         
        	wf3.append(get_warp_label(f1_3.permute(1,2,0), f2_3.permute(1,2,0), pf3[i].permute(1,2,0)))
        	wf4.append(get_warp_label(f1_4.permute(1,2,0), f2_4.permute(1,2,0), pf4[i].permute(1,2,0)))
        	wf5.append(get_warp_label(f1_5.permute(1,2,0), f2_5.permute(1,2,0), pf5[i].permute(1,2,0)))
        	wf6.append(get_warp_label(f1_5.permute(1,2,0), f2_5.permute(1,2,0), pf6[i].permute(1,2,0)))
        	wf7.append(get_warp_label(f1_7.permute(1,2,0), f2_7.permute(1,2,0), pf7[i].permute(1,2,0)))
        	wf8.append(get_warp_label(f1_7.permute(1,2,0), f2_7.permute(1,2,0), pf8[i].permute(1,2,0)))
         
        wf3 = torch.stack(wf3, dim=3).permute(3,2,0,1)
        wf4 = torch.stack(wf4, dim=3).permute(3,2,0,1)
        wf5 = torch.stack(wf5, dim=3).permute(3,2,0,1)
        wf6 = torch.stack(wf6, dim=3).permute(3,2,0,1)
        wf7 = torch.stack(wf7, dim=3).permute(3,2,0,1)
        wf8 = torch.stack(wf8, dim=3).permute(3,2,0,1)
        
        '''
        softmax = nn.Softmax().cuda()
        cur_pre_softmax = softmax(cur_prediction)
        warp_res =softmax(warp_prev_output)
        #l1_loss = loss_L1(cur_pre_softmax, warp_res)
        
        wfm3 = torch.argmax(wf3, dim = 1)
        wfm4 = torch.argmax(wf4, dim = 1)
        wfm5 = torch.argmax(wf5, dim = 1)
        wfm6 = torch.argmax(wf6, dim = 1)
        wfm7 = torch.argmax(wf7, dim = 1)
        wfm8 = torch.argmax(wf8, dim = 1)
        '''
        
        warp_output_max = torch.argmax(warp_prev_output, dim = 1)
        heatmaps = heatmaps[:,0,:,:]
        
        ce_loss = loss(cur_prediction, heatmaps)
        warp_output_loss = loss(cur_prediction, warp_output_max)
        wf3_loss = loss_l1(cf3, wf3)
        wf4_loss = loss_l1(cf4, wf4)
        wf5_loss = loss_l1(cf5, wf5)
        wf6_loss = loss_l1(cf6, wf6)
        wf7_loss = loss_l1(cf7, wf7)
        wf8_loss = loss_l1(cf8, wf8) 
        wf_loss = wf3_loss+wf4_loss+wf5_loss+wf6_loss+wf7_loss+wf8_loss

        loss_output = ce_loss + lamda * warp_output_loss + beta * wf_loss
                
        iou += helper.cal_iou(heatmaps, cur_prediction)
        metrics.append(loss_output.item())
        warp_metrics.append(warp_output_loss.item())
        warp_feature_metrics.append(wf_loss.item())

    iou /= len(data_loader)
    
    img = helper.make_validation_img(batch['cur'].numpy()[:,:,:,:],
                                    batch['label'].numpy(),
                                    cur_prediction.cpu().numpy())
    print("img.shape:",img.shape)
    cv2.imwrite('%s/validate_%d_%.4f.png'%(save_dir, epoch, iou),
                img[:, :, ::-1])
    

    end_time = time.time()
    metrics = np.asarray(metrics, np.float32)
    warp_metrics = np.asarray(warp_metrics, np.float32)
    warp_feature_metrics = np.asarray(warp_feature_metrics, np.float32)
    return metrics, warp_metrics, warp_feature_metrics, end_time - start_time, iou


if __name__ == '__main__':

    workers = 0
    batch_size = 16
    epochs = 100
    base_lr = 1e-3
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume', default='', help='resume training from checkpoint ...', type=str)
    parser.add_argument('-l', '--lamda', default=1.0, help='weight of output warp loss',type = float)
    parser.add_argument('-b', '--beta', default=1.0, help='weight of feature warp loss', type = float)
    args = parser.parse_args()
    lamda = args.lamda #2.0
    beta = args.beta #0.2
    print(f"lamda: {lamda}, beta: {beta}")
    
    save_dir = './checkpoints/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    net = network(2)
    
    #ckpt = torch.load('./ckpt_v6.030.ckpt')['state_dict']
    #net.load_from_img_pretrained(ckpt)

    loss_CE = nn.CrossEntropyLoss()
    loss_L1 = nn.L1Loss()

    ###########################################################
    # data_dir = './dataset/DAVIS'
    # davis_train = DAVIS(data_dir, 'training')
    # davis_val = DAVIS(data_dir, 'test')
    data_dir = '../dataset/youtube/'
    ytb_train = YOUTUBE(data_dir, 'training')
    ytb_val = YOUTUBE(data_dir, 'val')
    data_dir = '../dataset/versa/'
    versa_train = versa(data_dir, 'training')
    versa_val = versa(data_dir, 'val')

    #train_dataset = DataGenerator([versa_train], phase='train')
    train_dataset = DataGenerator([ytb_train, versa_train], phase='train')
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers)
    #val_dataset = DataGenerator([versa_val], phase='val')
    val_dataset = DataGenerator([ytb_val, versa_val], phase='val')
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=workers)
        
    print('Train sample number: %d' % len(train_dataset))
    print('Val sample number: %d' % len(val_dataset))
    ############################################################
    
    start_epoch = 1
    lr = base_lr
    best_val_loss = float('inf')
    log_mode = 'w'
    if os.path.exists(args.resume):
        print('loading checkpoint %s'%(args.resume))
        checkpoint = torch.load(args.resume)
        # start_epoch = checkpoint['epoch'] + 1
        # lr = checkpoint['lr']
        # best_val_loss = checkpoint['best_val_loss']
        net.load_state_dict(checkpoint['state_dict'])
        log_mode = 'a'

    net = net.cuda()
    loss_CE = loss_CE.cuda()
    loss_L1 = loss_L1.cuda()
    cudnn.benchmark = True
    # net = DataParallel(net)
    
    paras = []
    for i in range(3, 9):
        for j in range(1, 4):
            paras.extend(list(getattr(net, f'layer{i}_{j}').parameters()))
            
    optimizer = torch.optim.SGD(paras, lr, momentum=0.9, weight_decay=1e-4)

    # optimizer = torch.optim.SGD(
        # net.parameters(), lr, momentum=0.9, weight_decay=1e-4)


    lrs = LRScheduler(
        lr, patience=5, factor=0.5, min_lr=0.001 * lr, best_loss=best_val_loss)
    
    #with torch.no_grad():
    #   val_metrics, val_warpmetrics, val_feature_metrics, val_time, val_iou = validate(val_loader, net, loss_CE, loss_L1, lamda, beta, 0)

    for epoch in range(start_epoch, epochs + 1):
        train_metrics, train_warpmetrics, train_feature_metrics, train_time = train(train_loader, net, loss_CE, loss_L1, lamda, beta, optimizer, lr)
        with torch.no_grad():
            val_metrics, val_warpmetrics, val_feature_metrics, val_time, val_iou = validate(val_loader, net, loss_CE, loss_L1, lamda, beta, epoch)
        
        print_log(
            epoch,
            lr,
            train_metrics,
            train_warpmetrics,
            train_feature_metrics,
            train_time,
            val_metrics,
            val_warpmetrics,
            val_feature_metrics,
            val_time,
            val_iou,
            save_dir=save_dir,
            log_mode=log_mode)

        val_loss = np.mean(val_metrics)
        lr = lrs.update_by_rule(val_loss)
        if val_loss < best_val_loss or epoch % 1 == 0 or lr is None:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            state_dict = net.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()
            torch.save({
                'epoch': epoch,
                'save_dir': save_dir,
                'state_dict': state_dict,
                'lr': lr,
                'best_val_loss': best_val_loss
            }, os.path.join(save_dir, 'ckpt_%03d.ckpt' % epoch))

        if lr is None:
            print('Training is early-stopped')
            break
