import numpy as np
import cv2
import torch
import glob
import os
import time
import gzip
import pickle

#warping function of pytorch version on cuda by ASOP
#flow1 is the forward flow (H, W, 2)
#flow2 is the backward flow (H, W, 2)
#img is the input flow (H, W, n)
#warp_cur = warp(forward_flow, backward_flow, prev)
#warp_prev = warp(backward_flow, forward_flow, cur)

def warp(flow1, flow2, img): #H,W,C  h*w*2

    flow_shape = flow1.shape
    label_shape = img.shape  
    height, width = flow1.shape[0], flow1.shape[1]

    output = torch.zeros(label_shape).cuda(async=True)   #output = np.zeros_like(img, dtype=img.dtype)
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

    output[valid, :] = img.float()[dx[valid].long(), dy[valid].long(), :]
    
    return output #HW3  cuda

def main():
    prev = cv2.imread('00000.jpg')
    cur = cv2.imread('00005.jpg')
    flow1 = pickle.loads(gzip.GzipFile('forward_0_5.pkl', 'rb').read()) #'forward_0_5.pkl'
    flow2 = pickle.loads(gzip.GzipFile('backward_5_0.pkl', 'rb').read())  #'backward_5_0.pkl'
    print("read flow and image")
    #warp(forward, backward, 0th)  #  0 -> 1
    #warp(backward, forward, 1th)  #  1 -> 0
    flow1 = torch.from_numpy(flow1)
    flow2 = torch.from_numpy(flow2)
    cur = torch.from_numpy(cur)
    prev = torch.from_numpy(prev)
    w0 = warp(flow1,flow2,prev).numpy()  #0->1
    w1 = warp(flow2,flow1,cur).numpy()  #1->0
    print("finish warp")
    cv2.imwrite('warp_forward.png', w0)
    cv2.imwrite('warp_backward.png', w1)

if __name__ == '__main__':
    main()
