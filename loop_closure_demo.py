# -*- coding: utf8 -*-
"""
     OpenSeqSLAM
     Copyright 2013, Niko S��nderhauf Chemnitz University of Technology niko@etit.tu-chemnitz.de

     pySeqSLAM is an open source Python implementation of the original SeqSLAM algorithm published by Milford and Wyeth at ICRA12 [1]. SeqSLAM performs place recognition by matching sequences of images.

     [1] Michael Milford and Gordon F. Wyeth (2012). SeqSLAM: Visual Route-Based Navigation for Sunny Summer Days and Stormy Winter Nights. In Proc. of IEEE Intl. Conf. on Robotics and Automation (ICRA)

     gy_Rick:
     I change the demo.py, support loop closure and some visualizations

"""

from pyseqslam.parameters import defaultParameters
from pyseqslam.utils import AttributeDict
import matplotlib
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.io import loadmat, savemat
import time
import os
import numpy as np
from pyseqslam.seqslam import *

def main():

    # set the default parameters
    groundtruthPath = '../datasets/new_college/NewCollegeGroundTruth.mat'
    params_for_leftCamera = defaultParameters()
    params_for_rightCamera = defaultParameters()

    # set the custom parameters
    # config left camera

    ds = AttributeDict()
    ds.name = 'left_camera'
    ds.imagePath = '../datasets/new_college/images'
    if not os.path.exists(ds.imagePath):
        print('imagePath is not existed')
        return
    num_of_items = len(os.listdir(ds.imagePath))
    ds.prefix = ''
    ds.extension = '.jpg'
    ds.suffix = ''
    ds.dataFormat = '04'  # 00001
    ds.imageSkip = 2  # use every n-nth image
    ds.imageIndices = range(1, num_of_items+1, ds.imageSkip)
    ds.saveFile = '%s-%d-%d-%d' % (ds.name, ds.imageIndices[0], ds.imageSkip, ds.imageIndices[-1])

    ds.preprocessing = AttributeDict()
    ds.preprocessing.save = 1
    ds.preprocessing.load = 1
    # ds.crop=[1 1 60 32]  # x0 y0 x1 y1  cropping will be done AFTER resizing!
    ds.crop = []
    params_for_leftCamera.dataset = [ds, deepcopy(ds)]

    # config right camera
    ds1 = deepcopy(ds)
    ds1.name = 'right_camera'
    ds1.imageIndices = range(2, num_of_items+1, ds1.imageSkip)
    ds1.saveFile = '%s-%d-%d-%d' % (ds1.name, ds1.imageIndices[0], ds1.imageSkip, ds1.imageIndices[-1])
    params_for_rightCamera.dataset = [ds1, deepcopy(ds1)]

    # where to save / load the results
    params_for_leftCamera.savePath = './results16'
    params_for_rightCamera.savePath = './results16'
    if not os.path.exists(params_for_leftCamera.savePath):
        os.mkdir(params_for_leftCamera.savePath)

    # now process the dataset
    left = SeqSLAM(params_for_leftCamera)
    t1 = time.time()
    left_results = left.findLoopClosure()
    t2 = time.time()
    print("left camera time taken: " + str(t2 - t1))

    right = SeqSLAM(params_for_rightCamera)
    t1 = time.time()
    right_results = right.findLoopClosure()
    t2 = time.time()
    print("right camera time taken: " + str(t2 - t1))

    # process and visual result
    # 1. show loop closure
    thresh = 0.8
    row = left_results.matches.shape[0]
    if len(left_results.matches) > 0 and len(right_results.matches) > 0:
        left_idx = np.copy(left_results.matches[:, 0])  # The LARGER the score, the WEAKER the match.
        left_idx[left_results.matches[:, 1] > thresh] = np.nan  # remove the weakest matches

        right_idx = np.copy(right_results.matches[:, 0])  # The LARGER the score, the WEAKER the match.
        right_idx[right_results.matches[:, 1] > thresh] = np.nan  # remove the weakest matches

        # when left and right camera get loop closure simultaneously count
        # m = left_idx
        # m[left_idx != right_idx] = np.nan

    else:
        print("Zero matches")
        return

    # get loop matrix   x is current frame, y is reference frame
    loopMat = np.zeros((row, row))
    for i in range(row):
        if not np.isnan(left_idx[i]):
            loopMat[i, int(left_idx[i])] = 1
        if not np.isnan(right_idx[i]):
            loopMat[i, int(right_idx[i])] = 1

    groundtruthMat = loadmat(groundtruthPath)
    groundtruthMat = groundtruthMat['truth']
    leftM = groundtruthMat[::2, ::2]
    rightM = groundtruthMat[1::2, 1::2]
    groundtruthMat = leftM * rightM

    cross_set = loopMat * groundtruthMat
    TP = np.where(cross_set == 1)
    FP = loopMat - cross_set
    FP = np.where(FP)

    # fig 1 subplot loop matrix
    fig1, ax1 = plt.subplots(1, 2)
    ax1[0].scatter(TP[0], TP[1], 8, 'b')  # ideally, this would only be the diagonal
    ax1[0].scatter(FP[0], FP[1], 8, 'r')  # ideally, this would only be the diagonal
    ax1[0].set_title('Loop Closure')
    ax1[0].axis('square')
    ax1[0].legend(['TP','FP'])
    ax1[0].grid()

    idx = np.where(groundtruthMat == 1)
    ax1[1].scatter(idx[0], idx[1], 8, 'b')
    ax1[1].set_title('Ground Truth')
    ax1[1].axis('square')
    ax1[1].grid()


    # draw PR curve
    gt_loop = np.sum(groundtruthMat)
    thresh = [0.6 + 0.01*i for i in range(39)]
    pr = []
    if len(left_results.matches) > 0 and len(right_results.matches) > 0:
        for mu in thresh:
            left_idx = np.copy(left_results.matches[:, 0])  # The LARGER the score, the WEAKER the match.
            left_idx[left_results.matches[:, 1] > mu] = np.nan  # remove the weakest matches
            right_idx = np.copy(right_results.matches[:, 0])  # The LARGER the score, the WEAKER the match.
            right_idx[right_results.matches[:, 1] > mu] = np.nan  # remove the weakest matches
            # when left and right camera get loop closure simultaneously count

            loopMat = np.zeros((row, row))
            for i in range(row):
                if not np.isnan(left_idx[i]):
                    loopMat[i, int(left_idx[i])] = 1
                if not np.isnan(right_idx[i]):
                    loopMat[i, int(right_idx[i])] = 1

            p_loop = np.sum(loopMat)
            TP = np.sum(loopMat * groundtruthMat)
            pre = TP / p_loop
            rec = TP / gt_loop
            pr.append([pre, rec])

    pr = np.array(pr)
    fig2, ax2 = plt.subplots()
    ax2.plot(pr[:, 1], pr[:, 0], '-o')
    ax2.set_title('PR Curve')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.grid()






if __name__ == "__main__":
    main()
