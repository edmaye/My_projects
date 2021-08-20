import numpy as np
import cv2
import os,glob



stride=256+128
window=512

dataset_dir = 'dataset_0713'
# 0-background  1-green  2-house  3-land  4-ocean 5-road
map = [
    [0,1,2,3,5,-1],     # 这张图片里只出现了5类，因此最后一个空位补-1
    [0,1,2,4,5,-1],
    [0,1,2,3,4,5],
    [0,1,2,3,5,-1],
    [0,1,2,3,5,-1]
]
imgs = [os.path.join(dataset_dir,str(i)+'_json/img.png') for i in range(1,6)]
labels = [os.path.join(dataset_dir,str(i)+'_json/label.png') for i in range(1,6)]

if not os.path.exists(dataset_dir+'/train'):
    os.makedirs(dataset_dir+'/train')
if not os.path.exists(dataset_dir+'/label'):
    os.makedirs(dataset_dir+'/label')

for img_id, (img_path,label_path) in enumerate(zip(imgs,labels)):

    id = int(img_path.split('/')[-2].split('_')[0])
    print(id)
    img = cv2.imread(img_path)
    label = cv2.imread(label_path)

    mask = np.zeros((label.shape[0],label.shape[1],1))
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            color = label[i,j,:]

            if (color==[0,0,0]).all():
                tag = map[id-1][0]
            elif (color==[0,0,128]).all():
                tag = map[id-1][1]
            elif (color==[0,128,0]).all():
                tag = map[id-1][2]
            elif (color==[0,128,128]).all():
                tag = map[id-1][3]
            elif (color==[128,0,0]).all():
                tag = map[id-1][4]
            elif (color==[128,0,128]).all():
                #print('图片:',id,img_id)
                tag = map[id-1][5]
            mask[i,j] = tag


    shape = img.shape
    idx=0
    for i in np.arange(0,shape[0],stride):
        for j in np.arange(0,shape[1],stride):
            if(i+window > shape[0] or j+window > shape[1]):
                continue
            idx=idx+1
            cv2.imwrite(os.path.join(dataset_dir,'train',str(id)+'_'+str(idx)+'.png'),img[i:i+window,j:j+window])
            cv2.imwrite(os.path.join(dataset_dir,'label',str(id).split('.')[0]+'_'+str(idx)+'.png'),mask[i:i+window,j:j+window])
            


