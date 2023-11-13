import numpy as np
import os
from PIL import Image
import csv


"""
计算分割的性能

TP (True Positive)：真正例，即模型预测为正例，实际也为正例；
FP (False Positive)：假正例，即模型预测为正例，实际为反例；
FN (False Negative)：假反例，即模型预测为反例，实际为正例；
TN (True Negative)：真反例，即模型预测为反例，实际也为反例。
将这四种指标放在同一表格中，便构成了混淆矩阵(横着代表预测值，竖着代表真实值):
P\L     预测P    预测N
真实P      TP      FP
真实N      FN      TN
"""


# 计算混淆矩阵
def get_hist(label_true, label_pred, n_class):
    """
    label_true是转化为一维数组的真实标签，label_pred是转化为一维数组的预测结果，n_class是类别数
    hist是一个混淆矩阵(一个二维数组)，可以写成hist[label_true][label_pred]的形式
    """
    # mask在和label_true相对应的索引的位置上填入true或者false
    # label_true[mask]会把mask中索引为true的元素输出
    mask = (label_true >= 0) & (label_true < n_class)
    # n_class * label_true[mask].astype(int) + label_pred[mask]计算得到的是二维数组元素变成一位数组元素的时候的地址取值(每个元素大小为1)，返回的是一个numpy的list
    # np.bincount()会给出索引对应的元素个数
    hist = np.bincount(n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2)
    hist = hist.reshape(n_class, n_class)
    
    return hist


# precision（精确率）
def cal_Precision(hist):
    # precision = TP / TP + FP
    if hist[1][1] == 0:
        return 0
    precision = hist[1][1] / (hist[0][1] + hist[1][1])
    return precision


# recall（召回率）
def cal_Recall(hist):
    # recall = TP / TP + FN
    if hist[1][1] == 0:
        return 0
    recall = hist[1][1] / (hist[1][0] + hist[1][1])
    return recall


def cal_F1(precision, recall):
    if precision == 0 or recall == 0:
        return 0
    F1 = 2 * precision * recall / (precision + recall)
    return F1


def cal_IOU(hist):
    if hist[1][1] == 0:
        return 0
    IOU = hist[1][1] / (hist[0][1] + hist[1][0] + hist[1][1])
    return IOU


def cal_mIOU(hist):
    if hist[0][0] == 0 and hist[1][1] == 0:
        return 0
    elif hist[0][0] == 0:
        IOU1 = 0
        IOU2 = hist[1][1] / (hist[0][1] + hist[1][0] + hist[1][1])
        mIOU = (IOU1 + IOU2) / 2
    elif hist[1][1] == 0:
        IOU1 = hist[0][0] / (hist[0][0] + hist[0][1] + hist[1][0])
        IOU2 = 0
        mIOU = (IOU1 + IOU2) / 2
    else:
        IOU1 = hist[0][0] / (hist[0][0] + hist[0][1] + hist[1][0])
        IOU2 = hist[1][1] / (hist[0][1] + hist[1][0] + hist[1][1])
        mIOU = (IOU1 + IOU2) / 2
    return mIOU


def cal_results(result_path, mask_path, result_save_csv_path):
    num = 0
    total_IOU = 0
    total_mIOU = 0
    total_Precision = 0
    total_Recall = 0
    total_F1 = 0
    for file in os.listdir(mask_path):
        file_name = file.split('.')[0]

        result_img_name = file_name + '_d.jpg'
        result_img = np.array(Image.open(os.path.join(result_path, result_img_name)).convert('1'))

        result_mask_name = file_name + '_e.jpg'
        result_mask = np.array(Image.open(os.path.join(result_path, result_mask_name)).convert('1'))

        hist = get_hist(result_mask, result_img, 2)
        total_IOU += cal_IOU(hist)
        total_mIOU += cal_mIOU(hist)
        p = cal_Precision(hist)
        r = cal_Recall(hist)
        f = cal_F1(p, r)
        total_Precision += p
        total_Recall += r
        total_F1 += f
        num += 1

    IOU = total_IOU / num
    mIOU = total_mIOU / num
    Precision = total_Precision / num
    Recall = total_Recall / num
    F1 = total_F1 / num
    print("IoU: {}, MIoU: {}, Precision: {}, Recall: {}, F1-measure: {}".format(IOU, mIOU, Precision, Recall, F1))

    # 保存测试数据到csv文件
    item = {'IoU': str(IOU), 'MIoU': str(mIOU), 'Precision': str(Precision),
            'Recall': str(Recall), 'F1-measure': str(F1)}
    fieldnames = ['IoU', 'MIoU', 'Precision', 'Recall', 'F1-measure']
    save_result = os.path.join('.', result_save_csv_path)
    with open(save_result, mode='a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        # 判断表格内容是否为空
        if not os.path.getsize(save_result):
            writer.writeheader()  # 写入表头
        writer.writerows([item])