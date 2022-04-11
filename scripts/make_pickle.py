import numpy as np
import os
import cv2

PATH = './datasets/SR/NTIRE22-StereoSR/Train'

LR_FOLDER = 'LR_x4'
HR_FOLDER = 'HR'


lr_lists = []
hr_lists = []

cnt = 0

for idx in range(1, 801):

    L_name = f'{idx:04}_L.png'
    R_name = f'{idx:04}_R.png'


    LR_L = cv2.imread(os.path.join(PATH, LR_FOLDER, L_name))
    LR_R = cv2.imread(os.path.join(PATH, LR_FOLDER, R_name))

    HR_L = cv2.imread(os.path.join(PATH, HR_FOLDER, L_name))
    HR_R = cv2.imread(os.path.join(PATH, HR_FOLDER, R_name))

    LR = np.concatenate([LR_L, LR_R], axis=-1)
    HR = np.concatenate([HR_L, HR_R], axis=-1)

    lr_lists.append(LR)
    hr_lists.append(HR)

    cnt = cnt + 1
    if cnt % 50 == 0:
        print(f'cnt .. {cnt}, idx: {idx}')



import pickle
with open('./datasets/ntire-stereo-sr.train.lr.pickle', 'wb') as f:
    pickle.dump(lr_lists, f)

with open('./datasets/ntire-stereo-sr.train.hr.pickle', 'wb') as f:
    pickle.dump(hr_lists, f)



# print(f'... {lr_all_np.shape}, {lr_all_np.dtype}')
# print(f'... {hr_all_np.shape}, {hr_all_np.dtype}')

# np.save('./datasets/ntire-stereo-sr.train.lr.npy', lr_all_np)
# np.save('./datasets/ntire-stereo-sr.train.hr.npy', hr_all_np)


