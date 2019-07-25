import cv2
import numpy as np

def above(tpl,target):
    h, w = target.shape[:2]
    th, tw = tpl.shape[:2]
    #获得模板的宽高
    result = cv2.matchTemplate(target, tpl, cv2.TM_SQDIFF_NORMED)
    #matchTemplate(): image:待搜索的图像(大图)  temple:搜索模板  method:指定匹配方法, CV_TM_SQDIFF_NORMED:归一化平方差匹配法
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    #minMaxLoc寻找矩阵中最小值和最大值位置
    #min_Val参数表示返回的最小值，如果不需要，则使用NULL。
    #max_Val参数表示返回的最大值，如果不需要，则使用NULL。
    #min_Loc参数表示返回的最小位置的指针（在2D情况下）； 如果不需要，则使用NULL。
    #max_Loc参数表示返回的最大位置的指针（在2D情况下）； 如果不需要，则使用NULL。
    tl = min_loc
    if tl[0]<w/2 and tl[1]<h/2:
        a = int(tl[0] + 2.5 * tw)
        b = int(tl[0] + tw)
        c = int(tl[1] + 2.5 * th)
        d = int(tl[1] + th)
    elif tl[0]<w/2 and tl[1]>h/2:
        a = int(tl[0] + 2.5 * tw)
        b = int(tl[0] + tw)
        c = int(tl[1] - 2.5 * th)
        d = int(tl[1] - th)
    elif tl[0]>w/2 and tl[1]<h/2:
        a = int(tl[0] - 2.5 * tw)
        b = int(tl[0] - tw)
        c = int(tl[1] + 2.5 * th)
        d = int(tl[1] + th)
    else:
        a = int(tl[0] - 2.5 * tw)
        b = int(tl[0] - tw)
        c = int(tl[1] - 2.5 * th)
        d = int(tl[1] - th)
    crop = target[d:c, b:a]
    #剪裁源图片
    return crop

#tpl = cv2.imread("coin.jpg")
#target = cv2.imread("coin5.jpg")
#crop = above(tpl,target)
#cv2.imwrite("img3.jpg",crop)