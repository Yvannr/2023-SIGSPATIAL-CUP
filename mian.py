
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss,BCELoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from torchvision import transforms as T
from osgeo import gdal
import json
import torch.nn.functional as F
from PIL import Image
from PIL import ImageEnhance
from warmup_scheduler import GradualWarmupScheduler


def imgread(fileName):
    dataset = gdal.Open(fileName)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    data = dataset.ReadAsArray(0, 0, width, height)
    if len(data.shape) == 3:
        data = data.swapaxes(1, 0).swapaxes(1, 2)
    return data


def truncated_linear_stretch(image, truncated_value, max_out=255, min_out=0):
    def gray_process(gray):
        truncated_down = np.percentile(gray, truncated_value)
        truncated_up = np.percentile(gray, 100 - truncated_value)
        gray = (gray - truncated_down) / (truncated_up - truncated_down) * (max_out - min_out) + min_out
        gray = np.clip(gray, min_out, max_out)
        gray = np.uint8(gray)
        return gray

    image_stretch = []
    for i in range(image.shape[2]):
        # 只拉伸RGB
        if (i < 3):
            gray = gray_process(image[:, :, i])
        else:
            gray = image[:, :, i]
        image_stretch.append(gray)
    image_stretch = np.array(image_stretch)
    image_stretch = image_stretch.swapaxes(1, 0).swapaxes(1, 2)
    return image_stretch


def randomColor(image): #随机颜色
    """
    对图像进行颜色抖动
    :param image: PIL的图像image
    :return: 有颜色色差的图像image
    """
    image = Image.fromarray(image)
    # image = Image.open(os.path.join(root_path, img_name))
    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因子
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因子
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    sharpness_image = ImageEnhance.Sharpness(contrast_image).enhance(random_factor)
    return np.array(sharpness_image)


def DataAugmentation(image, label, mode):
    if (mode == "train"):
        # pass
        hor = random.choice([True, False])
        if (hor):
            #  图像水平翻转
            image = np.flip(image, axis=1)
            label = np.flip(label, axis=1)
        ver = random.choice([True, False])
        if (ver):
            #  图像垂直翻转
            image = np.flip(image, axis=0)
            label = np.flip(label, axis=0)

        rot90 = random.choice([True, False])
        if(rot90):
            image = np.rot90(image)
            label = np.rot90(label)
        # radcrop = RandomCrop()
        stretch = random.choice([True, False])
        # if (stretch):
        #     image = truncated_linear_stretch(image, 0.5)
        # image = randomColor(image)
        # image = RandomCrop(image, label)

        # ColorAug = random.choice([True, False])

    return image, label


def get_one_hot(label, N):
    size = list(label.size())
    label1 = label.view(-1)  # reshape 为向量
    ones = torch.sparse.torch.eye(N).cuda()
    ones = ones.index_select(0, label1)   # 用上面的办法转为换one hot
    size.append(N)  # 把类别输目添到size的尾后，准备reshape回原来的尺寸
    return ones.view(*size)


def Dice_loss(inputs, target, beta=1, smooth=1e-5):
    n, c, h, w = inputs.size()
    target = get_one_hot(target, c)
    nt, ht, wt, ct = target.size()

    assert target.size() == target.size(), "the size of predict and target must be equal."
    num = target.size(0)

    pre = torch.sigmoid(inputs).view(num, -1)
    tar = target.view(num, -1)

    intersection = (pre * tar).sum(-1).sum()  # 利用预测值与标签相乘当作交集
    union = (pre + tar).sum(-1).sum()

    dice_loss = 1 - 2 * (intersection + smooth) / (union + smooth)

    return dice_loss


def CoMa(inputs, target, beta=1, smooth=1e-5, threhold=0.5):
    n, c, h, w = inputs.size()
    target_oh = get_one_hot(target, c)
    target_v = target.view(n, -1)
    nt, ht, wt, ct = target_oh.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target_oh.view(n, -1, ct).cuda()

    temp_squ = torch.argmax(temp_inputs,-1)
    diff = torch.sum((temp_squ - target_v) == 0)
    size_squeeze = temp_squ.size(0)*temp_squ.size(1)
    oa = diff/size_squeeze

    temp_inputs = torch.nn.functional.one_hot(temp_squ, num_classes = c)
    tp = torch.sum(temp_target * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
    fn = torch.sum(temp_target, axis=[0, 1]) - tp
    tn = 512 * 512 * 2 - (tp + fp + fn)

    se = float(torch.sum(tp)) / (float(torch.sum(tp + fn)) + 1e-5)
    pc = float(torch.sum(tp)) / (float(torch.sum(tp + fp)) + 1e-5)

    F1 = 2 * se * pc / (se + pc + 1e-5)
    acc = torch.mean(tp / (tp + fp + smooth))

    iou = tp / (fp + fn + tp + smooth)
    iou_ground = iou[-1]
    iou_noground = iou[:-1]
    miou = torch.mean(iou_noground)

    return tp, fp, fn, oa.item()


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_lake_probs(data_root, temperature):
    with open(os.path.join(data_root, 'sample_class_stats.json'), 'r') as of:
        sample_class_stats = json.load(of)
    overall_class_stats = {}
    for s in sample_class_stats:
        s.pop('file')
        for c, n in s.items():
            c = int(c)
            if c not in overall_class_stats:
                overall_class_stats[c] = n
            else:
                overall_class_stats[c] += n
    overall_class_stats = {
        k: v
        for k, v in sorted(
            overall_class_stats.items(), key=lambda item: item[1])
    }
    freq = torch.tensor(list(overall_class_stats.values()))
    freq = freq / torch.sum(freq)
    freq = 1 - freq
    freq = torch.softmax(freq / temperature, dim=-1)

    return list(overall_class_stats.keys()), freq.numpy()


class TUDataset(Dataset):
    def __init__(self, image_paths, label_paths, mode, data_root):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.mode = mode
        self.len = len(image_paths)
        self.as_tensor = T.Compose([
            T.ToTensor(),
        ])
        self.rcs_class_temp = 8
        self.rcs_classes, self.rcs_classprob = get_lake_probs(
            data_root, self.rcs_class_temp)
        with open(
                os.path.join(data_root, 'samples_with_class.json'), 'r') as of:
            samples_with_class_and_n = json.load(of)
        samples_with_class_and_n = {
            int(k): v
            for k, v in samples_with_class_and_n.items()
            if int(k) in self.rcs_classes
        }
        self.samples_with_class = {}
        for c in self.rcs_classes:
            self.samples_with_class[c] = []
            for file, pixels in samples_with_class_and_n[c]:
                if pixels > 500:
                    self.samples_with_class[c].append(file.split('/')[-1])
            assert len(self.samples_with_class[c]) > 0

    def __len__(self):
        return self.len

    def __getitem__(self, index):

        if self.mode == "train":
            c = np.random.choice(self.rcs_classes, p=np.array([0.3, 0.7]))
            f1 = np.random.choice(self.samples_with_class[c])
            image = imgread(os.path.join(os.path.dirname(self.image_paths[0]), f1))
            label = imgread(os.path.join(os.path.dirname(self.label_paths[0]), f1))
            image, label = DataAugmentation(image, label, self.mode)
            image_array = np.ascontiguousarray(image)
            return self.as_tensor(image_array), label.astype(np.int64)
        elif self.mode == "val":
            image = imgread(self.image_paths[index])

            label = imgread(self.label_paths[index])

            image_array = np.ascontiguousarray(image)
            return self.as_tensor(image_array), label.astype(np.int64)


def trainn(num_classes, model, train_image_paths, val_image_paths, epoch_step, epoch_step_val, train_label_paths,
           val_label_paths, epoch, optimizer, batch_size):

    lr_scheduler_l = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
    lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=10, total_epoch=9, after_scheduler=lr_scheduler_l)

    ce_loss = CrossEntropyLoss()
    dice_loss = Dice_loss
    model = model.cuda()
    iter_num = 0
    dataroot = '/dataroot'
    db_train = TUDataset(train_image_paths, train_label_paths, mode='train', data_root=dataroot)
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
    db_train_val = TUDataset(val_image_paths, val_label_paths, mode='val', data_root=dataroot)
    valloader = DataLoader(db_train_val, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
    for epoch_num in range(epoch):
        vvacc = 0
        vvoa = 0
        vvloss = 0
        ttoa = 0
        ttloss = 0
        model.train()
        trep = epoch_step - epoch_step_val

        atp = torch.zeros((num_classes,), dtype=torch.float64).cuda()
        afp = torch.zeros((num_classes,), dtype=torch.float64).cuda()
        afn = torch.zeros((num_classes,), dtype=torch.float64).cuda()

        btp = torch.zeros((num_classes,), dtype=torch.float64).cuda()
        bfp = torch.zeros((num_classes,), dtype=torch.float64).cuda()
        bfn = torch.zeros((num_classes,), dtype=torch.float64).cuda()

        with tqdm(total=trep, desc=f'Epoch {epoch_num + 1}/{epoch}', postfix=dict, mininterval=0.3, colour='cyan') as pbar:

                for i, sampled in enumerate(trainloader):
                    image_batch, label_batch = sampled
                    del sampled
                    image_batch = image_batch.cuda()
                    label_batch = label_batch.cuda()
                    outputs = model(image_batch)

                    loss_ce = ce_loss(outputs, label_batch[:].long())

                    loss_dice = dice_loss(outputs, label_batch)
                    loss = 0.5 * loss_ce + 0.5 * loss_dice
                    ttloss += loss.item()

                    with torch.no_grad():
                        tp, fp, fn, oa = CoMa(outputs, label_batch)
                    atp += tp
                    afp += fp
                    afn += fn
                    ttoa += oa

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    rec = atp/(atp + afn + (1e-5))
                    pre = atp/(atp + afp + 1e-5)
                    F1 = 2 * rec * pre / (rec + pre + 1e-5)
                    iou = atp / (afp + afn + atp + 1e-5)
                    iter_num = iter_num + 1
                    pbar.set_postfix(**{'lr': get_lr(optimizer),
                                        'oa': ttoa / (i + 1),
                                        'Iou': np.round(np.array(iou.tolist()), 4).tolist(),
                                        'loss': ttloss / (i+1),
                                        'f1': np.round(np.array(F1.tolist()), 4).tolist()
                                        })
                    pbar.update(1)

        if (epoch_num+1) % 10 == 0:
            model.eval()
            with tqdm(total=epoch_step_val, desc=f'Epoch {epoch_num + 1}/{epoch}', postfix=dict, mininterval=0.3) as pbar:
                for j, vsampled in enumerate(valloader):
                    vimage_batch, vlabel_batch = vsampled
                    vimage_batch = vimage_batch.cuda()
                    vlabel_batch = vlabel_batch.cuda()
                    with torch.no_grad():
                        outputss = model(vimage_batch).cuda()
                        vloss_ce = ce_loss(outputss, vlabel_batch[:].long())
                        vloss_dice = dice_loss(outputss, vlabel_batch)
                        vloss = 0.5 * vloss_ce + 0.5 * vloss_dice
                        tp, fp, fn, oa = CoMa(outputss, vlabel_batch)
                        btp += tp
                        bfp += fp
                        bfn += fn
                        vvoa += oa
                        rec = btp / (btp + bfn + 1e-5)
                        pre = btp / (btp + bfp + 1e-5)
                        F1 = 2 * rec * pre / (rec + pre + 1e-5)
                        iou = btp / (bfp + bfn + btp + 1e-5)
                        vvloss += vloss.item()
                    pbar.set_postfix(**{'lr': get_lr(optimizer),
                                        'oa': vvoa / (j+1),
                                        'Iou': np.round(np.array(iou.tolist()), 4).tolist(),
                                        # 'acc': vvacc / (j + 1),
                                        'f1': np.round(np.array(F1.tolist()), 4).tolist(),
                                        "loss": vvloss/(j+1)
                                        })
                    pbar.update(1)

            if (epoch_num + 1) % 10 == 0:
                torch.save(model.state_dict(), 'savemodel/ep%03d-loss%.3f-acc%.3f.pth' % ((epoch_num + 1), vvloss / (epoch_step_val), np.round(np.array(F1.tolist()), 4).tolist()[-1]))
        lr_scheduler.step()
