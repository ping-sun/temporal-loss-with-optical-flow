import numpy as np


def cal_iou(heatmaps, prediction):
    pred_heat = (prediction[:, 1] > prediction[:, 0]).long()
    equal = (pred_heat == heatmaps).long()
    inter = (equal * heatmaps).sum().item()
    union = heatmaps.size(0) * heatmaps.size(1) * heatmaps.size(2) - \
        (equal * (1 - heatmaps)).sum().item()
    if union==0:
        union = 1
    return inter / union


def make_validation_img(img, lab, pre):
    print("shape:", img.shape, lab.shape, pre.shape)
    # img (image): (np.array) batchsize * 3 * H * W
    # lab (label): (np.array) batchsize * H * W
    # pre (predition): (np.array) batchsize * H * W
    # shape: (9, 3, 512, 512) (9, 3, 512, 512) (9, 2, 512, 512)
    
    # img
    img *= 255
    img = img.astype(np.uint8)
    img = np.concatenate(img, axis=1)
    #print("imge.shape:", img.shape)  #3 4608 512
    # label
    lab = lab[:,0,:,:]
    lab = np.concatenate(lab)
    #print("label.shape:", lab.shape) #27 512 512
    lab = np.array([i * lab for i in img]).transpose(1, 2, 0)
    #print("label.shape:", lab.shape)
    # predict
    pre = pre[:, 1] > pre[:, 0]
    pre = np.concatenate(pre)
    pre = np.array([i * pre for i in img]).transpose(1, 2, 0)
    #print("pred.shape:", pre.shape)

    img = img.transpose(1, 2, 0)
    #print("res.shape:",img.shape)
    return np.concatenate([img, lab, pre], 1)
