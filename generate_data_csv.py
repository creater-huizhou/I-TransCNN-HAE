import os
import csv
import numpy as np


# 建立csv文件保存训练集和测试集路径
def creat_train_csv(input_directory, img_dir, mask_dir, output_directory, csv_name):
    # 如果csv文件不存在
    if os.path.exists(os.path.join(input_directory, csv_name)):
        os.remove(os.path.join(input_directory, csv_name))
    image_list = []
    # 遍历文件夹，获得所有的图片的路径
    for name in os.listdir(os.path.join(input_directory, img_dir)):
        image_list.append(os.path.join(input_directory, img_dir, name))

    # 随机打散顺序
    np.random.shuffle(image_list)
    # 创建csv文件，并存储图片路径及其label信息
    with open(os.path.join(output_directory, csv_name), mode='w', newline='')as f:
        writer = csv.writer(f)
        for img in image_list:
            name = str(np.random.randint(1, 201)) + '.jpg'
            label = os.path.join(input_directory, mask_dir, name)
            writer.writerow([img, label])
        print('written into csv file:', csv_name)


def creat_test_csv(input_directory, img_dir, mask_dir, output_directory, csv_name):
    # 如果csv文件不存在
    if not os.path.exists(os.path.join(input_directory, csv_name)):
        image_list = []
        # 遍历文件夹，获得所有的图片的路径
        for name in os.listdir(os.path.join(input_directory, img_dir)):
            image_list.append(os.path.join(input_directory, img_dir, name))

        # 随机打散顺序
        np.random.shuffle(image_list)
        # 创建csv文件，并存储图片路径及其label信息
        with open(os.path.join(output_directory, csv_name), mode='w', newline='')as f:
            writer = csv.writer(f)
            for img in image_list:
                name = img.split('/')[-1]
                # print(name)
                label = os.path.join(input_directory, mask_dir, name)
                # print(label)
                writer.writerow([img, label])
            print('written into csv file:', csv_name)