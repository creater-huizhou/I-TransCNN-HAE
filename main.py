import torch
from torch.autograd import Variable
import os
import csv
from model import CMT_TransCNN
from loss import cal_loss
from dataset import My_train_Dataset
from torch.utils.data import DataLoader
from generate_data_csv import creat_train_csv


def train(model, max_epochs, batchs, save_dir_path):
    model.train()

    min_train_loss = 10000
    epochs_without_improvement = 0
    patience = 3
    for epoch in range(11, max_epochs):
        if epoch < 20:
            learning_rate = 0.0005
        elif epoch < 50:
            learning_rate = 0.0002
        else:
            learning_rate = 0.0001
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.0, 0.9))

        # 建立csv文件保存训练集路径
        input_path = '.'
        output_path = '.'
        creat_train_csv(input_path, 'train_imgs', 'train_masks', output_path, 'train_data.csv')
        # 准备数据集
        train_dataset = My_train_Dataset('.', 'train_data.csv')
        # 把数据集装载到DataLoader里
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batchs)

        num = 0
        total_loss = 0
        total_l1_loss = 0
        total_perceptual_loss = 0
        total_style_loss = 0
        for step, (img_contaminated, img_truth, img_path, mask_path) in enumerate(train_loader):  # 遍历训练集
            # 构建梯度记录器
            img_contaminated = Variable(img_contaminated)
            img_contaminated = img_contaminated.to(device)
            img_truth = img_truth.to(device)
            # 前向计算获得重建的图片
            img_pred = model(img_contaminated)
            # 计算损失
            loss, l1_loss, perceptual_loss, style_loss = cal_loss(img_pred, img_truth)
            total_loss += loss
            total_l1_loss += l1_loss
            total_perceptual_loss += perceptual_loss
            total_style_loss += style_loss
            num = num + 1
            # 更新梯度等参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                loss = loss.cpu().detach().numpy()
                l1_loss = l1_loss.cpu().detach().numpy()
                perceptual_loss = perceptual_loss.cpu().detach().numpy()
                style_loss = style_loss.cpu().detach().numpy()
                print("epoch: {:>3d}, step: {:>4d}, total_loss: {:.6f}, l1_loss: {:.6f}, perceptual_loss: {:.6f}, "
                      "style_loss: {:.6f}".format(epoch, step, loss, l1_loss, perceptual_loss, style_loss))
        
        train_loss = (total_loss / num).cpu().detach().numpy()
        train_l1_loss = (total_l1_loss / num).cpu().detach().numpy()
        train_perceptual_loss = (total_perceptual_loss / num).cpu().detach().numpy()
        train_style_loss = (total_style_loss / num).cpu().detach().numpy()

        print("epoch: {:>3d}, total_loss: {:.6f}, l1_loss: {:.6f}, perceptual_loss: {:.6f}, style_loss: {:.6f}".format(
            epoch, train_loss, train_l1_loss, train_perceptual_loss, train_style_loss))

        # 保存loss数据到csv文件
        item = {'epoch': str(epoch), 'total_loss': str(train_loss), 'l1_loss': str(train_l1_loss),
                'perceptual_loss': str(train_perceptual_loss), 'style_loss': str(train_style_loss)}
        fieldnames = ['epoch', 'total_loss', 'l1_loss', 'perceptual_loss', 'style_loss']
        save_loss = os.path.join('.', 'loss.csv')
        with open(save_loss, mode='a', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            # 判断表格内容是否为空，如果为空就添加表头
            if not os.path.getsize(save_loss):
                writer.writeheader()  # 写入表头
            writer.writerows([item])
                    
        if train_loss < min_train_loss:
            # 模型保存
            save_para_name = 'model-epoch-' + str(epoch) + '.pt'
            save_para_path = os.path.join(save_dir_path, save_para_name)
            torch.save({'params': model.state_dict()}, save_para_path)
            min_train_loss = train_loss
        else:
            epochs_without_improvement = epochs_without_improvement + 1
                
        # 如果验证集上的损失连续patience个epoch没有提高，则停止训练
        if epochs_without_improvement == patience:
            print('Early stopping at epoch {}...'.format(epoch+1))
            break
    
    # img_name = os.path.join('./', 'loss.png')
    # draw_train_process('train_loss', 'epoch', 'loss', epoch_list, train_loss_list, 'train_loss', img_name)
    

if __name__ == "__main__":
    batch_size = 4
    max_epochs = 100

    save_dir_path = './model_save'
    # 如果文件夹不存在，就创建文件夹
    if not os.path.exists(save_dir_path):
        os.mkdir(save_dir_path)

    # 查看GPU是否可用
    if torch.cuda.is_available():
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CMT_TransCNN(device=DEVICE, batch=batch_size).to(device)
    model_para = torch.load('./model_save/model-epoch-10.pt')
    model.load_state_dict(model_para['params'])
    train(model, max_epochs, batch_size, save_dir_path)