# coding: utf-8
import mxnet as mx
from mtcnn_detector import MtcnnDetector
import cv2
import os
import time
import numpy as np 
import cv2
import dlib

hat_img = cv2.imread("hat2.png",-1)
img = cv2.imread('oscar1.jpg')
detector = MtcnnDetector(model_folder='model', ctx=mx.cpu(0), num_worker = 4 , accurate_landmark = True)
results = detector.detect_face(img)

# 给img中的人头像加上圣诞帽，人脸最好为正脸
def add_hat(img,hat_img):
    # 分离rgba通道，合成rgb三通道帽子图，a通道后面做mask用
    r,g,b,a = cv2.split(hat_img) 
    rgb_hat = cv2.merge((r,g,b))
    cv2.imwrite("hat_alpha.jpg",a)
    if results is not None:
        total_boxes = results[0]
        i = 0
        for d in total_boxes:

            #x,y,w,h = d.left(),d.top(), d.right()-d.left(), d.bottom()-d.top()
            x,y,w,h = int(d[0]),int(d[1]), int(d[2])-int(d[0]), int(d[3])-int(d[1])
            #shape = predictor(img, d)
            shape = results[1][i]
            print(len(shape))
            # 选取左右眼眼角的点
            point1 = shape[0:2]
            point2 = shape[4:6]

            # 求两点中心
            eyes_center = ((int(point1[0]+point2[0])//2),int((point1[1]+point2[1])//2))
            print(eyes_center)
            # cv2.circle(img,eyes_center,3,color=(0,255,0))  
            # cv2.imshow("image",img)
            # cv2.waitKey()

            #  根据人脸大小调整帽子大小
            factor = 1.5
            resized_hat_h = int(round(rgb_hat.shape[0]*w/rgb_hat.shape[1]*factor))
            resized_hat_w = int(round(rgb_hat.shape[1]*w/rgb_hat.shape[1]*factor))

            if resized_hat_h > y:
                resized_hat_h = y-1

            # 根据人脸大小调整帽子大小
            resized_hat = cv2.resize(rgb_hat,(resized_hat_w,resized_hat_h))

            # 用alpha通道作为mask
            mask = cv2.resize(a,(resized_hat_w,resized_hat_h))
            mask_inv =  cv2.bitwise_not(mask)

            # 帽子相对与人脸框上线的偏移量
            dh = 0
            dw = 0
            # 原图ROI
            # bg_roi = img[y+dh-resized_hat_h:y+dh, x+dw:x+dw+resized_hat_w]
            bg_roi = img[y+dh-resized_hat_h:y+dh,(eyes_center[0]-resized_hat_w//3):(eyes_center[0]+resized_hat_w//3*2)]

            # 原图ROI中提取放帽子的区域
            bg_roi = bg_roi.astype(float)
            mask_inv = cv2.merge((mask_inv,mask_inv,mask_inv))
            alpha = mask_inv.astype(float)/255

            # 相乘之前保证两者大小一致（可能会由于四舍五入原因不一致）
            alpha = cv2.resize(alpha,(bg_roi.shape[1],bg_roi.shape[0]))
            # print("alpha size: ",alpha.shape)
            # print("bg_roi size: ",bg_roi.shape)
            bg = cv2.multiply(alpha, bg_roi)
            bg = bg.astype('uint8')

            cv2.imwrite("bg.jpg",bg)
            # cv2.imshow("image",img)
            # cv2.waitKey()

            # 提取帽子区域
            hat = cv2.bitwise_and(resized_hat,resized_hat,mask = mask)
            cv2.imwrite("hat.jpg",hat)
            
            # cv2.imshow("hat",hat)  
            # cv2.imshow("bg",bg)

            # print("bg size: ",bg.shape)
            # print("hat size: ",hat.shape)

            # 相加之前保证两者大小一致（可能会由于四舍五入原因不一致）
            hat = cv2.resize(hat,(bg_roi.shape[1],bg_roi.shape[0]))
            # 两个ROI区域相加
            add_hat = cv2.add(bg,hat)
            # cv2.imshow("add_hat",add_hat) 

            # 把添加好帽子的区域放回原图
            img[y+dh-resized_hat_h:y+dh,(eyes_center[0]-resized_hat_w//3):(eyes_center[0]+resized_hat_w//3*2)] = add_hat

            # 展示效果
            # cv2.imshow("img",img )  
            # cv2.waitKey(0)
            i = i + 1
            print(i)

        return img
# 读取帽子图，第二个参数-1表示读取为rgba通道，否则为rgb通道

output = add_hat(img,hat_img)
cv2.imwrite("output.jpg",output)