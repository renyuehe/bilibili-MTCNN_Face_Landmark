# 创建数据集

from torch.utils.data import Dataset
import os
import numpy as np
import torch
from PIL import Image

# 数据集
class FaceDataset(Dataset):
    def __init__(self,path, is_train:bool):
        # 初始化路径
        self.path = path

        # 数据集
        self.dataset = []

        # 将（正样本、负样本、部分样本） 的标签都加入到 数据集中
        # 854669
        # 1598395
        # 1368326
        if is_train:
            self.dataset.extend(open(os.path.join(path,"positive.txt")).readlines()[:5]) # 打开正样本标签文档，逐行读取，再添加至列表中
            self.dataset.extend(open(os.path.join(path,"negative.txt")).readlines()[:10])
            self.dataset.extend(open(os.path.join(path,"part.txt")).readlines()[:5])
        else:
            self.dataset.extend(open(os.path.join(path,"positive.txt")).readlines()[5:6]) # 打开正样本标签文档，逐行读取，再添加至列表中
            self.dataset.extend(open(os.path.join(path,"negative.txt")).readlines()[10:12])
            self.dataset.extend(open(os.path.join(path,"part.txt")).readlines()[5:6])

    def __len__(self):
        return len(self.dataset) # 数据集长度

    def __getitem__(self, index): # 获取数据
        strs = self.dataset[index].strip().split(" ") # 取一条数据，去掉前后字符串，再按空格分割

        #标签：置信度+偏移量
        cond = torch.Tensor([int(strs[1])]) # []莫丢，否则指定的是shape
        offset = torch.Tensor([float(strs[2]),float(strs[3]),float(strs[4]),float(strs[5])])
        landmark_offset = torch.Tensor([float(strs[6]),float(strs[7]),
                           float(strs[8]),float(strs[9]),
                           float(strs[10]),float(strs[11]),
                           float(strs[12]),float(strs[13]),
                           float(strs[14]),float(strs[15])])

        #样本：img_data
        img_path = os.path.join(self.path,strs[0]) # 图片绝对路径
        img_data = torch.Tensor(np.array(Image.open(img_path))/255.-0.5)  # 打开-->array-->归一化去均值化-->转成tensor
        img_data = img_data.permute(2,0,1) # CHW

        # print(img_data.shape) # WHC
        # a = img_data.permute(2,0,1) #轴变换
        # print(a.shape) #[3, 48, 48]：CWH

        # 返回 图片tensor, 置信度, 偏移量, landmark_偏移量
        return img_data, cond, offset, landmark_offset

# 测试
if __name__ == '__main__':
    path = r"D:\Desktop\Celeba2\celeba_3\48" # 只以尺寸为48的为例
    train_dataset = FaceDataset(path, is_train=True)
    # test_dataset = FaceDataset(path, is_train=False)

    print(train_dataset[0])
    print(train_dataset[3])
    print(train_dataset[4][0].shape)
    print(train_dataset[4][1].shape)
    print(train_dataset[4][2].shape)
    print(train_dataset[4][3].shape)

    # print(dataset[0][0].shape) # 图片 img_data
    # print(dataset[1][1].shape) # 置信度
    # print(dataset[2][2].shape)       # 偏移量






