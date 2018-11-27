import numpy as np
import cv2
import os
import gzip
import pickle

#warping function of numpy version by ASOP
#flow1 is the forward flow (H, W, 2)
#flow2 is the backward flow (H, W, 2)
#img is the input flow (H, W, n)
#warp_cur = warp(forward_flow, backward_flow, prev)
#warp_prev = warp(backward_flow, forward_flow, cur)

def warp(flow1, flow2, img):
    output = np.zeros_like(img, dtype=img.dtype)
    height = flow1.shape[0]
    width = flow1.shape[1]
    flow_t = np.zeros_like(flow1, dtype=flow1.dtype)

    grid = np.indices((height, width)).swapaxes(0, 1).swapaxes(1, 2)
    dx = grid[:, :, 0] + flow2[:, :, 1]
    dy = grid[:, :, 1] + flow2[:, :, 0]
    sx = np.floor(dx).astype(int)
    sy = np.floor(dy).astype(int)
    valid = (sx >= 0) & (sx < height - 1) & (sy >= 0) & (sy < width - 1)

    sx_mat = np.dstack((sx, sx + 1, sx, sx + 1)).clip(0, height - 1)
    sy_mat = np.dstack((sy, sy, sy + 1, sy + 1)).clip(0, width - 1)
    sxsy_mat = np.abs((1 - np.abs(sx_mat - dx[:, :, np.newaxis])) *
                      (1 - np.abs(sy_mat - dy[:, :, np.newaxis])))

    for i in range(4):
        flow_t = flow_t + sxsy_mat[:, :, i][:, :, np.
                                            newaxis] * flow1[sx_mat[:, :, i],
                                                             sy_mat[:, :, i], :]

    valid = valid & (np.linalg.norm(
        flow_t[:, :, [1, 0]] + np.dstack((dx, dy)) - grid, axis=2) < 100)

    flow_t = (flow2 - flow_t) / 2.0
    dx = grid[:, :, 0] + flow_t[:, :, 1]
    dy = grid[:, :, 1] + flow_t[:, :, 0]

    valid = valid & (dx >= 0) & (dx < height - 1) & (dy >= 0) & (dy < width - 1)
    output[valid, :] = img[dx[valid].round().astype(int), dy[valid].round()
                              .astype(int), :]
    return output

def main():
    prev = cv2.imread('00000.jpg')
    cur = cv2.imread('00005.jpg')
    flow1 = pickle.loads(gzip.GzipFile('forward_0_5.pkl', 'rb').read()) #'forward_0_5.pkl'
    flow2 = pickle.loads(gzip.GzipFile('backward_5_0.pkl', 'rb').read())  #'backward_5_0.pkl'
    print("read flow and image")
    #warp(forward, backward, 0th)  #  0 -> 1
    #warp(backward, forward, 1th)  #  1 -> 0

    w0 = warp(flow1,flow2,prev)  #0->1
    w1 = warp(flow2,flow1,cur) #1->0
    print("finish warp")
    cv2.imwrite('warp_forward.png', w0)
    cv2.imwrite('warp_backward.png', w1)

if __name__ == '__main__':
    main()
