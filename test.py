import torch
from model import CMT_TransCNN
from dataset import My_test_Dataset
from torch.utils.data import DataLoader
from torchvision import utils
from calculation_indicator import *
from generate_data_csv import creat_test_csv
import cv2


def generate_norm_map(input_img, output_img, mask_img, img_paths, result_path):
    # 保存图像
    dir_name = result_path
    # 如果文件夹不存在，就创建文件夹
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    imgname = img_paths.split('/')[-1]
    name = imgname.split('.')[0]
    # 原图
    path1 = name + '_a.jpg'
    input_img = (input_img + 1) / 2
    utils.save_image(input_img, os.path.join(dir_name, path1))
    # 预测图像
    path2 = name + '_b.jpg'
    output_img = (output_img + 1) / 2
    utils.save_image(output_img, os.path.join(dir_name, path2))
    # 异常图
    path3 = name + '_c.jpg'
    norm = input_img - output_img
    norm = torch.abs(norm)
    utils.save_image(norm, os.path.join(dir_name, path3))
    #
    path4 = name + '_d.jpg'
    norm = norm.cpu().detach().numpy()
    norm = 0.299 * norm[0] + 0.587 * norm[1] + 0.114 * norm[2]
    norm = norm.squeeze()
    norm = cv2.medianBlur(norm, 5)
    norm = cv2.dilate(norm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    norm_mean = np.mean(norm)
    norm_std = np.std(norm)
    threshold = norm_mean + 2 * norm_std
    norm[norm < threshold] = int(0)
    norm[norm >= threshold] = int(255)
    norm = Image.fromarray(norm).convert('L')
    norm.save(os.path.join(dir_name, path4))
    # mask
    path5 = name + '_e.jpg'
    utils.save_image(mask_img, os.path.join(dir_name, path5))
    print("{} is finished!".format(imgname))


def test(result_path, mask_paths, result_save_csv_path):
    if not os.path.exists(result_path):
        for step, (img_contaminated, img_truth, img_path, mask_path) in enumerate(test_loader):
            img_contaminated = img_contaminated.to(device)
            img_truth = img_truth.to(device)
            # 前向计算获得重建的图片
            img_pred = model(img_contaminated)

            output_img = img_pred[0, :, :, :].squeeze()
            input_img = img_contaminated[0, :, :, :].squeeze()
            mask_img = img_truth[0, :, :, :].squeeze()
            img_paths = img_path[0]
            generate_norm_map(input_img, output_img, mask_img, img_paths, result_path)

    cal_results(result_path, mask_paths, result_save_csv_path)


# 建立csv文件保存训练集和测试集路径
input_path = '.'
output_path = '.'
creat_test_csv(input_path, 'test_imgs', 'test_masks', output_path, 'test_data.csv')

batch_size = 1
# 准备数据集
test_dataset = My_test_Dataset('.', 'test_data.csv')
# 把数据集装载到DataLoader里
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)
# 查看GPU是否可用
if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CMT_TransCNN(device=DEVICE, batch=batch_size).to(device)
model_para = torch.load('./model_save/model-epoch-11.pt')
# model_para = torch.load('./model_save/model-epoch-17.pt', map_location='cpu')
model.load_state_dict(model_para['params'])
model.eval()

# 测试输出图像保存路径
result_path = './results-epoch-11'
# 测试集mask路径
mask_path = './test_masks'
result_save_csv_path = 'epoch-11-test-result.csv'
test(result_path, mask_path, result_save_csv_path)
