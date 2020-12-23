import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import numpy as np
from PIL import Image
from pyseqslam.parameters import defaultParameters
import os
import time

fig, ax = plt.subplots(2, 4, figsize=(6.4*4,4.8*2))
gs = ax[0,0].get_gridspec()
for each in ax[:, 2:].flatten():
    each.remove()
axgps = fig.add_subplot(gs[:, 2:])
axgps.set_title('GPS Trajectory')
plt.show()
imgPath = '../datasets/new_college/Images/'
num_of_img = len(os.listdir(imgPath)) // 2

# load the match
left_match_path = './results16/LoopClosure-left_camera-1-2-2145-left_camera-1-2-2145.mat'
right_match_path = './results16/LoopClosure-right_camera-2-2-2146-right_camera-2-2-2146.mat'

left_match = loadmat(left_match_path)
right_match = loadmat(right_match_path)

groundtruthPath = '../datasets/new_college/NewCollegeGroundTruth.mat'
groundtruthMat = loadmat(groundtruthPath)
groundtruthMat = groundtruthMat['truth']
leftM = groundtruthMat[::2, ::2]
rightM = groundtruthMat[1::2, 1::2]

gpsPath = '../datasets/new_college/ImageCollectionCoordinates.mat'
gps = loadmat(gpsPath)
gps = gps['GPS']
gps_half = (gps[::2, :] + gps[1::2, :]) / 2

thresh = 0.8
left_idx = np.copy(left_match['Loop'][:, 0])
left_idx[left_match['Loop'][:, 1] > thresh] = np.nan
right_idx = np.copy(right_match['Loop'][:, 0])
right_idx[right_match['Loop'][:, 1] > thresh] = np.nan

red = []
blue = []
for i in range(num_of_img):

    ax[0,1].clear()
    ax[1,1].clear()
    ax[0,0].clear()
    ax[1,0].clear()
    axgps.clear()

    left_img = Image.open(imgPath + '%04d.jpg' % (2 * i + 1))
    right_img = Image.open(imgPath + '%04d.jpg' % (2 * i + 2))
    ax[0,0].imshow(left_img)
    ax[0,0].set_title('Left Camera')

    ax[1, 0].imshow(right_img)
    ax[1, 0].set_title('Right Camera')
    axgps.set_title('GPS Trajectory')

    if not np.isnan(left_idx[i]):
        left_loopimg = Image.open(imgPath + '%04d.jpg' % (2 * int(left_idx[i])+1))
        ax[0,1].imshow(left_loopimg)
        # check if it is correct
        if leftM[i, int(left_idx[i])] == 1:
            ax[0,1].set_title('True')
        else:
            ax[0, 1].set_title('False')
    else:
        ax[0,1].set_title('New Place')

    if not np.isnan(right_idx[i]):
        right_loopimg = Image.open(imgPath + '%04d.jpg' % (2 * int(right_idx[i])+2))
        ax[1,1].imshow(right_loopimg)
        # check if it is correct
        if rightM[i, int(right_idx[i])] == 1:
            ax[1,1].set_title('True')
        else:
            ax[1,1].set_title('False')
    else:
        ax[1,1].set_title('New Place')


    if np.isnan(left_idx[i]) and np.isnan(right_idx[i]):
        if len(blue) == 0:
            blue = np.array([[gps_half[i, 0], gps_half[i, 1]]])
        else:
            blue = np.concatenate((blue, np.array([[gps_half[i, 0], gps_half[i, 1]]])), axis=0)
        axgps.scatter(blue[:, 0], blue[:, 1], 12, 'b')
        if len(red) != 0:
            axgps.scatter(red[:, 0], red[:, 1], 12, 'r')
        plt.pause(0.1)
        continue

    if len(red) == 0:
        red = np.array([[gps_half[i, 0], gps_half[i, 1]]])
    else:
        red = np.concatenate((red, np.array([[gps_half[i, 0], gps_half[i, 1]]])), axis=0)
    axgps.scatter(red[:, 0], red[:, 1], 12, 'r')
    if len(blue) != 0:
        axgps.scatter(blue[:, 0], blue[:, 1], 12, 'b')

    plt.pause(0.5)



