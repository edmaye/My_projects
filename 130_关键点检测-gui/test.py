import cv2
import time
import numpy as np
from tkinter import *
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
import os


# COCO Output Format
keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']

#为每个关键点命名，一共18个，不包含背景
#鼻子-0, 脖子-1，右肩-2，右肘-3，右手腕-4，左肩-5，左肘-6，左手腕-7，右臀-8，右膝盖-9，右脚踝-10，左臀-11，左膝盖-12，左脚踝-13，右眼-14，左眼-15，右耳朵-16，左耳朵-17

POSE_PAIRS = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7], #脖子-右肩-右肘-右手腕  脖子-左肩-左肘-左手腕
              [1,8], [8,9], [9,10], [1,11], [11,12], [12,13], #脖子-右臀—右膝盖-右脚踝  脖子—左臀—左膝盖-左脚踝
              [1,0], [0,14], [14,16], [0,15], [15,17],  #脖子-右眼-右耳朵  左眼—左耳朵
              [2,17], [5,16] ]  #右肩—左耳朵 左肩—右耳朵

# index of pafs correspoding to the POSE_PAIRS
# e.g for POSE_PAIR(1,2), the PAFs are located at indices (31,32) of output, Similarly, (1,5) -> (39,40) and so on.
mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44],
          [19,20], [21,22], [23,24], [25,26], [27,28], [29,30],
          [47,48], [49,50], [53,54], [51,52], [55,56],
          [37,38], [45,46]]

colors = [[0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255],
         [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255],
         [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,0]]





class Detector():
    def __init__(self,protoFile="models/pose/coco/pose_deploy_linevec.prototxt",weightFile="models/pose/coco/pose_iter_440000.caffemodel",nPoints = 18):
        self.nPoints = nPoints     
        self.net = cv2.dnn.readNetFromCaffe(protoFile, weightFile)
    
    #关键点检测
    def getKeypoints(self,probMap, threshold=0.1):
        mapSmooth = cv2.GaussianBlur(probMap,(3,3),0,0)
        mapMask = np.uint8(mapSmooth>threshold)
        keypoints = []

        # 1. 找出对应于关键点的所有区域的轮廓(contours)
        contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # for each blob find the maxima
        # 对于每个关键点轮廓区域，找到最大值.
        for cnt in contours:
            # 2. 创建关键点的 mask；
            blobMask = np.zeros(mapMask.shape)
            blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
            # 3. 提取关键点区域的 probMap
            maskedProbMap = mapSmooth * blobMask
            # 4. 提取关键点区域的局部最大值.
            _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
            keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

        return keypoints


    # Find valid connections between the different joints of a all persons present
    def getValidPairs(self,output):
        valid_pairs = []
        invalid_pairs = []
        n_interp_samples = 10
        paf_score_th = 0.1
        conf_th = 0.7
        # loop for every POSE_PAIR
        for k in range(len(mapIdx)):
            # A->B constitute a limb
            pafA = output[0, mapIdx[k][0], :, :]
            pafB = output[0, mapIdx[k][1], :, :]
            pafA = cv2.resize(pafA, (self.frameWidth, self.frameHeight))
            pafB = cv2.resize(pafB, (self.frameWidth, self.frameHeight))

            # 检测第一个 limb 和第二个 limb 的关键点位置
            candA = self.detected_keypoints[POSE_PAIRS[k][0]]
            candB = self.detected_keypoints[POSE_PAIRS[k][1]]
            nA = len(candA)
            nB = len(candB)

            # 如果检测到 joint-pair 的关键点位置，则，
            # 	检查 candA 和 candB 中每个 joint.
            # 计算两个 joints 之间的距离向量(distance vector).
            # 计算两个 joints 之间插值点集合的 PAF 值.
            # 根据上述公式，计算 score 值，判断连接的有效性.

            if( nA != 0 and nB != 0):
                valid_pair = np.zeros((0,3))
                for i in range(nA):
                    max_j=-1
                    maxScore = -1
                    found = 0
                    for j in range(nB):
                        # Find d_ij
                        d_ij = np.subtract(candB[j][:2], candA[i][:2])
                        norm = np.linalg.norm(d_ij)
                        if norm:
                            d_ij = d_ij / norm
                        else:
                            continue
                        # Find p(u)
                        interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                                np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                        # Find L(p(u))
                        paf_interp = []
                        for k in range(len(interp_coord)):
                            paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                            pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))] ])
                        # Find E
                        paf_scores = np.dot(paf_interp, d_ij)
                        avg_paf_score = sum(paf_scores)/len(paf_scores)

                        # 判断连接是否有效.
                        # 如果对应于 PAF 的插值向量值大于阈值，则连接有效.
                        if ( len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples ) > conf_th :
                            if avg_paf_score > maxScore:
                                max_j = j
                                maxScore = avg_paf_score
                                found = 1
                    # Append the connection to the list
                    if found:
                        valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)

                # Append the detected connections to the global list
                valid_pairs.append(valid_pair)
            else: # If no keypoints are detected
                print("No Connection : k = {}".format(k))
                invalid_pairs.append(k)
                valid_pairs.append([])
        return valid_pairs, invalid_pairs


    # This function creates a list of keypoints belonging to each person
    # For each detected valid pair, it assigns the joint(s) to a person
    def getPersonwiseKeypoints(self,valid_pairs, invalid_pairs):
        # the last number in each row is the overall score
        personwiseKeypoints = -1 * np.ones((0, 19))

        for k in range(len(mapIdx)):
            if k not in invalid_pairs:
                partAs = valid_pairs[k][:,0]
                partBs = valid_pairs[k][:,1]
                indexA, indexB = np.array(POSE_PAIRS[k])

                for i in range(len(valid_pairs[k])):
                    found = 0
                    person_idx = -1
                    for j in range(len(personwiseKeypoints)):
                        if personwiseKeypoints[j][indexA] == partAs[i]:
                            person_idx = j
                            found = 1
                            break

                    if found:
                        personwiseKeypoints[person_idx][indexB] = partBs[i]
                        personwiseKeypoints[person_idx][-1] += self.keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][2]

                    # if find no partA in the subset, create a new subset
                    elif not found and k < 17:
                        row = -1 * np.ones(19)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        # add the keypoint_scores for the two keypoints and the paf_score
                        row[-1] = sum(self.keypoints_list[valid_pairs[k][i,:2].astype(int), 2]) + valid_pairs[k][i][2]
                        personwiseKeypoints = np.vstack([personwiseKeypoints, row])
        return personwiseKeypoints


    def detect(self,image1):
        self.frameWidth = image1.shape[1]
        self.frameHeight = image1.shape[0]
        # Fix the input Height and get the width according to the Aspect Ratio
        inHeight = 368
        inWidth = int((inHeight/self.frameHeight)*self.frameWidth)
        inpBlob = cv2.dnn.blobFromImage(image1, 1.0 / 255, (inWidth, inHeight),
                                (0, 0, 0), swapRB=False, crop=False)

        self.net.setInput(inpBlob)
        output = self.net.forward()    #根据长宽比，固定网路输入的 height，计算网络输入的 width.
  

        self.detected_keypoints = []
        self.keypoints_list = np.zeros((0,3))
        keypoint_id = 0
        threshold = 0.1

        for part in range(self.nPoints):
            probMap = output[0,part,:,:]
            probMap = cv2.resize(probMap, (image1.shape[1], image1.shape[0]))
            keypoints = self.getKeypoints(probMap, threshold)
            print("Keypoints - {} : {}".format(keypointsMapping[part], keypoints))
            keypoints_with_id = []
            for i in range(len(keypoints)):
                keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                self.keypoints_list = np.vstack([self.keypoints_list, keypoints[i]])
                keypoint_id += 1

            self.detected_keypoints.append(keypoints_with_id)


        frameClone_1 = image1.copy()
        for i in range(self.nPoints):
            for j in range(len(self.detected_keypoints[i])):
                cv2.circle(frameClone_1, self.detected_keypoints[i][j][0:2], 5, colors[i], -1, cv2.LINE_AA)
        img1 = frameClone_1
        #cv2.imshow("Keypoints",frameClone)



        valid_pairs, invalid_pairs = self.getValidPairs(output)
        personwiseKeypoints = self.getPersonwiseKeypoints(valid_pairs, invalid_pairs)
        for i in range(17):
            for n in range(len(personwiseKeypoints)):
                index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
                if -1 in index:
                    continue
                B = np.int32(self.keypoints_list[index.astype(int), 0])
                A = np.int32(self.keypoints_list[index.astype(int), 1])
                cv2.line(image1, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)


        return img1,image1
 

if __name__=='__main__':
    detector = Detector()
    img = cv2.imread('data/COCO_val2014_000000000241.jpg')
    detector.detect(img)