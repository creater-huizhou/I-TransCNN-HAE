import torchvision.models as models
import torch
import torch.nn as nn


class VGG19(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class ContenLoss(nn.Module):
    def __init__(self):
        super(ContenLoss, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)
        return G

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        perceptual_loss = 0
        style_loss = 0
        for i in range(len(x_vgg)):
            perceptual_loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())

        style_loss += self.criterion(self.compute_gram(x_vgg[3]), self.compute_gram(y_vgg[3]))
        style_loss += self.criterion(self.compute_gram(x_vgg[4]), self.compute_gram(y_vgg[4]))
        return perceptual_loss, style_loss


# 计算loss
def cal_loss(img_pred, img_truth):    
    l1_loss = nn.L1Loss()
    content_loss = ContenLoss()
    # 损失权重
    L1_LOSS_WEIGHT = 10
    PERCEPTUAL_LOSS_WEIGHT = 10
    STYLE_LOSS_WEIGHT = 250

    # l1 loss
    l1_loss = l1_loss(img_pred, img_truth)
    # perceptual_loss + style_loss
    perceptual_loss, style_loss = content_loss(img_pred, img_truth)
    loss = L1_LOSS_WEIGHT * l1_loss + PERCEPTUAL_LOSS_WEIGHT * perceptual_loss + STYLE_LOSS_WEIGHT * style_loss
            
    return loss, l1_loss, perceptual_loss, style_loss