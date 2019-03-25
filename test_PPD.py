#!/usr/bin/python2 -utt
# -*- coding: utf-8 -*-
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import time
import os
import cv2
import math
import numpy as np
from matplotlib import pyplot as plt


class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm, self).__init__()
        self.eps = 1e-10

    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim=1) + self.eps)
        x = x / norm.unsqueeze(-1).expand_as(x)
        return x

def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size= kernel_size, stride= stride, padding= pad, dilation=dilation, bias= False),
                         nn.BatchNorm2d(out_planes)
                         )
class PPD(nn.Module):
    """HardNet model definition
    """

    def __init__(self):
        super(PPD, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),

        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
        )

        self.branch1 = nn.Sequential(
            nn.AvgPool2d((8, 8), stride=(8, 8)),
            convbn(32, 16, 1, 1, 0, 1),
            nn.ReLU(inplace=True),
        )

        self.branch2 = nn.Sequential(
            nn.AvgPool2d((4, 4), stride=(4, 4)),
            convbn(32, 16, 1, 1, 0, 1),
            nn.ReLU(inplace=True),
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d((2, 2), stride=(2, 2)),
            convbn(32, 16, 1, 1, 0, 1),
            nn.ReLU(inplace=True),
        )

        self.branch4 = nn.Sequential(
            nn.AvgPool2d((1, 1), stride=(1, 1)),
            convbn(32, 16, 1, 1, 0, 1),
            nn.ReLU(inplace=True),
        )

        self.feat = nn.Sequential(
            nn.Dropout(0.3),
            nn.Conv2d(128, 128, kernel_size=8, bias=False),
            nn.BatchNorm2d(128, affine=False),
        )

        return

    def input_norm(self, x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(
            -1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward(self, input):

        out_put1 = self.layer1(self.input_norm(input))

        out_branch1 = self.branch1(out_put1)
        out_branch1 = F.upsample(out_branch1, (out_put1.size()[2], out_put1.size()[3]), mode='bilinear', align_corners=True)

        out_branch2 = self.branch2(out_put1)
        out_branch2 = F.upsample(out_branch2, (out_put1.size()[2], out_put1.size()[3]), mode='bilinear', align_corners=True)

        out_branch3 = self.branch3(out_put1)
        out_branch3 = F.upsample(out_branch3, (out_put1.size()[2], out_put1.size()[3]), mode='bilinear', align_corners=True)

        out_branch4 = self.branch4(out_put1)
        out_branch4 = F.upsample(out_branch4, (out_put1.size()[2], out_put1.size()[3]), mode='bilinear', align_corners=True)

        out_feature = torch.cat((out_branch4, out_branch3, out_branch2, out_branch1),1)


        out_put2 = self.layer2(out_feature)
        out_put3 = self.layer3(out_put2)
        x_feat = self.feat(out_put3)
        x = x_feat.view(x_feat.size(0), -1)

        return L2Norm()(x)

def describe_ppdnet(model, img, kpts, N, mag_factor, use_gpu = True):
    """
    Rectifies patches around openCV keypoints, and returns patches tensor
    """
    patches = []
    for kp in kpts:
        x,y = kp.pt
        s = kp.size
        a = kp.angle

        s = mag_factor * s / N
        cos = math.cos(a * math.pi / 180.0)
        sin = math.sin(a * math.pi / 180.0)

        M = np.matrix([
            [+s * cos, -s * sin, (-s * cos + s * sin) * N / 2.0 + x],
            [+s * sin, +s * cos, (-s * sin - s * cos) * N / 2.0 + y]])

        patch = cv2.warpAffine(img, M, (N, N),
                             flags=cv2.WARP_INVERSE_MAP + \
                             cv2.INTER_CUBIC + cv2.WARP_FILL_OUTLIERS)

        patches.append(patch)
    n_patches = len(patches)
    bs = 256
    descrs = np.zeros((n_patches, 128))
    for i in range(0, n_patches, bs):
        # compute output

        data = patches[i:i+bs]

        data = torch.from_numpy(np.asarray(data)).float() / 255.
        data -= 0.443728476019
        data /= 0.20197947209

        data = torch.unsqueeze(data, 1)

        if use_gpu:
            data = data.cuda()
        data = Variable(data)
        with torch.no_grad():
            out = model(data)
            descrs[i:i+bs,:] = out.detach().cpu().numpy()
    return np.float32(descrs)

if __name__ == '__main__':

    # load hardnet model
    DO_CUDA = True
    model_weights = './pretrained-models/PPD.pth'
    model = PPD()
    checkpoint = torch.load(model_weights)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    if DO_CUDA:
        model.cuda()
        print('Extracting on GPU')
    else:
        print('Extracting on CPU')
        model = model.cpu()

    # img1 = cv2.imread('imgs/Aerial/group1/00059.jpg', 0)
    # img2 = cv2.imread('imgs/Aerial/group1/00060.jpg', 0)
    #
    #
    # img1 = cv2.imread('imgs/Aerial/group2/00050.jpg', 0)
    # img2 = cv2.imread('imgs/Aerial/group2/00051.jpg', 0)
    #
    img1 = cv2.imread('data/Aerial/group3/00003.jpg', 0)
    img2 = cv2.imread('data/Aerial/group3/00004.jpg', 0)




    brisk = cv2.BRISK_create()
    kp1, des1 = brisk.detectAndCompute(img1, None)
    kp2, des2 = brisk.detectAndCompute(img2, None)

    mag_factor = 3.5

    start = time.time()
    desc_tfeat1 = describe_ppdnet(model, img1, kp1, 32, mag_factor)
    desc_tfeat2 = describe_ppdnet(model, img2, kp2, 32, mag_factor)
    end = time.time()
    print("PPD BRISK TIME: %f" % (end - start))


    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(desc_tfeat1, desc_tfeat2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)

    print("PPD-before: length of good pairs: ", len(good))

    # img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, 0, flags=2)
    draw_params = dict(matchColor=(255, 0, 0),  # draw matches in green color
                       singlePointColor=None,
                       flags=2)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

    # plt.imshow(img3), plt.show()
    plt.imsave("PPD_before.jpg", img3)

    scr_sift_pt = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_sift_pt = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(scr_sift_pt, dst_sift_pt, cv2.RANSAC, 10)
    matchesMask = mask.ravel().tolist()

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
    print("PPD-after: length of good pairs: ", sum(matchesMask))
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

    # plt.imshow(img3), plt.show()
    plt.imsave("PPD_after.jpg", img3)
    #

####################################################################################

    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1_sift = sift.detectAndCompute(img1, None)
    kp2, des2_sift = sift.detectAndCompute(img2, None)
    mag_factor = 7.5

    start = time.time()
    desc_tfeat1 = describe_ppdnet(model, img1, kp1, 32, mag_factor)
    desc_tfeat2 = describe_ppdnet(model, img2, kp2, 32, mag_factor)
    end = time.time()
    print("PPD SIFT TIME: %f" % (end - start))


    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(desc_tfeat1, desc_tfeat2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)

    print("PPD-before: length of good pairs: ", len(good))

    # img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, 0, flags=2)
    draw_params = dict(matchColor=(255, 0, 0),  # draw matches in green color
                       singlePointColor=None,
                       flags=2)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

    # plt.imshow(img3), plt.show()
    plt.imsave("SIFT-PPD_before.jpg", img3)

    scr_sift_pt = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_sift_pt = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(scr_sift_pt, dst_sift_pt, cv2.RANSAC, 10.0)
    matchesMask = mask.ravel().tolist()

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
    print("PPD-after: length of good pairs: ", sum(matchesMask))
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

    # plt.imshow(img3), plt.show()
    plt.imsave("SIFT-PPD_after.jpg", img3)