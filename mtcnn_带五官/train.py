import time
# 创建训练器----以训练三个网络
import os
from torch.utils.data import DataLoader
import torch
from torch import nn
import torch.optim as optim
from face_dataset import FaceDataset # 导入数据集
from sklearn.metrics import r2_score
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split

# 创建训练器
class Trainer:
    def __init__(self, net, save_path, dataset_path, isCuda=True):
        '''
        创建训练器,对 P、R、O 网络通用的训练器
        Args:
            net: 网络
            save_path:  参数保存路径
            dataset_path: 训练数据路劲
            isCuda: cuda加速为True
        '''
        self.net = net
        self.save_path = save_path
        self.dataset_path = dataset_path
        self.isCuda = isCuda
        self.sumerWriter = SummaryWriter("tensorboard")

        if self.isCuda:      # 默认后面有个else
            self.net.cuda()  # 给网络加速

        # 创建损失函数
        # 置信度损失
        self.cls_loss_fn = nn.BCELoss() # ★二分类交叉熵损失函数，是多分类交叉熵（CrossEntropyLoss）的一个特例；用BCELoss前面必须用sigmoid激活,用CrossEntropyLoss前面必须用softmax函数
        # 偏移量损失
        self.offset_loss_fn = nn.MSELoss()

        # 创建优化器
        self.optimizer = optim.Adam(self.net.parameters())
        # self.optimizer = optim.SGD(self.net.parameters, lr=0.01)

        # 恢复网络训练---加载模型参数，继续训练
        if os.path.exists(self.save_path): # 如果文件存在，接着继续训练
            net.load_state_dict(torch.load(self.save_path))

    # 训练方法
    def train(self):
        print("加载数据集>>start:", time.time())


        train_faceDataset = FaceDataset(self.dataset_path, True) # 训练数据集
        test_faceDataset = FaceDataset(self.dataset_path, False) # 测试集合


        train_dataloader = DataLoader(train_faceDataset, batch_size=5, shuffle=True, num_workers=1,drop_last=True) # 数据加载器
        test_dataloader = DataLoader(test_faceDataset, batch_size=5, shuffle=True, num_workers=1, drop_last=True)  # 测试集 数据加载器

        #num_workers=4：有4个线程在加载数据(加载数据需要时间，以防空置)；drop_last：为True时表示，防止批次不足报错。
        print("加载数据集>>end:",time.time())

        num = 0
        while True:
            # 训练集
            for i, (img_data_, category_, offset_, landmark_offset_) in enumerate(train_dataloader): # 样本，置信度，偏移量
                self.net.train()
                if self.isCuda:                    # cuda把数据读到显存里去了(先经过内存)；没有cuda在内存，有cuda在显存
                    img_data_ = img_data_.cuda()  # [N, 3, 12, 12]
                    category_ = category_.cuda() # N, 1]

                    offset_ = offset_.cuda()    # [N, 4]
                    landmark_offset_ = landmark_offset_.cuda() # [N, 10]

                # 网络输出，因为P、R、O网络的输入都是 NCHW 结构，所以这里可以直接输入
                _output_category, _output_offset, _output_landmark_offset = self.net(img_data_) # 输出置信度，偏移量
                # print(_output_category.shape)     # [N, 1, 1, 1]
                # print(_output_offset.shape)       # [N, 4, 1, 1]
                # print(_output_landmark_offset.shape) # [N, 10, 1, 1]
                # P网络的结果是 N111 和 N411，R和O的结果是 N1和 N4，为了让P网络的结果和 R和O网络统一，这里需要shape
                output_category = _output_category.reshape(-1, 1) # [N,1]
                output_offset = _output_offset.reshape(-1, 4)     # [N,4]
                _output_landmark_offset = _output_landmark_offset.reshape(-1,10) # [N,10]
                # output_landmark = _output_landmark.view(-1, 10)


                # 计算分类的损失----置信度
                # category_ 和 output_category 的形状是相同的,所以他们可以通过同样的 mask 获得对应位置的置信度
                # masked_select 函数返回不会保持形状，但是没有关系，因为取出来的 category 和 output_category 一定是对应的关系，可直接用来做损失
                category_mask = torch.lt(category_, 2)  # 对置信度小于2的正样本（1）和负样本（0）进行掩码; ★部分样本（2）不参与损失计算；符合条件的返回1，不符合条件的返回0
                category = torch.masked_select(category_, category_mask)              # 对“标签”中置信度小于2的选择掩码，返回符合条件的结果
                output_category = torch.masked_select(output_category, category_mask) # 预测的“标签”进掩码，返回符合条件的结果

                train_cls_loss = self.cls_loss_fn(output_category, category)                # 对置信度做损失


                # 计算bound回归的损失----偏移量
                # nonzero 返回的是 NV 结构, N代表有多少个 非0 元素，V代表每一个非0元素的索引（V的大小等于offset的维度）
                offset_mask = torch.gt(category_, 0)  # 对置信pytorc度大于0的标签（1）正样本（2）部分样本，进行掩码；★负样本不参与计算,负样本没偏移量;[N,1]
                if torch.sum(offset_mask) == 0:
                    continue
                offset_index = torch.nonzero(offset_mask)[:, 0]  # 选出非负样本的索引；[244]

                offset = offset_[offset_index]                   # 标签里的偏移量；[244,4]
                landmark_offset = landmark_offset_[offset_index] # 标签里的landmark偏移量；[244,10]??

                output_offset = output_offset[offset_index]      # 输出的偏移量；[244,4]
                output_landmark_offset = _output_landmark_offset[offset_index] # 输出的landmark偏移量；[244,10] ??

                train_offset_loss = self.offset_loss_fn(output_offset, offset)  # 偏移量损失
                train_landmark_offset_loss = self.offset_loss_fn(output_landmark_offset, landmark_offset)

                #总损失
                train_loss = train_cls_loss + train_offset_loss + train_landmark_offset_loss

                # 反向传播，优化网络
                self.optimizer.zero_grad() # 清空之前的梯度
                train_loss.backward()           # 计算梯度
                self.optimizer.step()    # 优化网络

                # 可解释方差
                # score = explained_variance_score(offset.detach().cpu(), output_offset.detach().cpu())
                # r2 得分
                score = r2_score(offset.detach().cpu(), output_offset.detach().cpu())
                landmark_score = r2_score(landmark_offset.detach().cpu(), output_landmark_offset.detach().cpu())

                print("train r2_score >> ", score)
                print("train landmark_r2_score >> ", landmark_score)

                #输出损失：loss-->gpu-->cup（变量）-->tensor-->array
                print("i=", i, "train_loss:", train_loss.cpu().data.numpy(), " train_cls_loss:", train_cls_loss.cpu().data.numpy(), " train_offset_loss",
                      train_offset_loss.cpu().data.numpy())

                # 保存
                if (i+1) % 2 == 0:
                    print("save success ------------------------------------->>",self.save_path)
                    torch.save(self.net.state_dict(), self.save_path) # state_dict保存网络参数，save_path参数保存路径 # 每轮次保存一次；最好做一判断：损失下降时保存一次

                    # 测试集
                    for i, (img_data_, category_, offset_, landmark_offset_) in enumerate(test_dataloader):  # 样本，置信度，偏移量
                        self.net.eval()
                        if self.isCuda:  # cuda把数据读到显存里去了(先经过内存)；没有cuda在内存，有cuda在显存
                            img_data_ = img_data_.cuda()  # [N, 3, 12, 12]
                            category_ = category_.cuda()  # N, 1]

                            offset_ = offset_.cuda()  # [N, 4]
                            landmark_offset_ = landmark_offset_.cuda() # [N, 10]

                        # 网络输出，因为P、R、O网络的输入都是 NCHW 结构，所以这里可以直接输入
                        _output_category, _output_offset, _output_landmark_offset = self.net(img_data_)  # 输出置信度，偏移量
                        # print(_output_category.shape)     # [N, 1, 1, 1]
                        # print(_output_offset.shape)       # [N, 4, 1, 1]
                        # P网络的结果是 N111 和 N411，R和O的结果是 N1和 N4，为了让P网络的结果和 R和O网络统一，这里需要shape
                        output_category = _output_category.reshape(-1, 1)  # [N,1]
                        output_offset = _output_offset.reshape(-1, 4)  # [N,4]
                        output_landmark_offset = _output_landmark_offset.reshape(-1, 4) # [N,4]

                        # 计算分类的损失----置信度
                        # category_ 和 output_category 的形状是相同的,所以他们可以通过同样的 mask 获得对应位置的置信度
                        # masked_select 函数返回不会保持形状，但是没有关系，因为取出来的 category 和 output_category 一定是对应的关系，可直接用来做损失
                        category_mask = torch.lt(category_, 2)  # 对置信度小于2的正样本（1）和负样本（0）进行掩码; ★部分样本（2）不参与损失计算；符合条件的返回1，不符合条件的返回0
                        category = torch.masked_select(category_, category_mask)  # 对“标签”中置信度小于2的选择掩码，返回符合条件的结果
                        output_category = torch.masked_select(output_category, category_mask)  # 预测的“标签”进掩码，返回符合条件的结果
                        test_cls_loss = self.cls_loss_fn(output_category, category)  # 对置信度做损失

                        # 计算bound回归的损失----偏移量
                        # nonzero 返回的是 NV 结构, N代表有多少个 非0 元素，V代表每一个非0元素的索引（V的大小等于offset的维度）
                        offset_mask = torch.gt(category_, 0)  # 对置信pytorc度大于0的标签（1）正样本（2）部分样本，进行掩码；★负样本不参与计算,负样本没偏移量;[N,1]
                        offset_index = torch.nonzero(offset_mask)[:, 0]  # 选出非负样本的索引；[244]

                        offset = offset_[offset_index]  # 标签里的偏移量；[244,4]
                        landmark_offset = landmark_offset_[offset_index] # 标签里的 landmark偏移率；[244, 10]

                        output_offset = output_offset[offset_index]  # 输出的偏移量；[244,4]
                        output_landmark_offset = output_landmark_offset[offset_index] # 输出的偏移率；[244,10]
                        test_offset_loss = self.offset_loss_fn(output_offset, offset)  # 偏移量损失
                        test_landmark_offset_loss = self.offset_loss_fn(output_landmark_offset, landmark_offset) # landmark偏移率损失

                        # 可解释方差
                        # score = explained_variance_score(offset.detach().cpu(), output_offset.detach().cpu())
                        # r2 得分
                        score = r2_score(offset.detach().cpu(), output_offset.detach().cpu())
                        landmark_score = r2_score(landmark_offset.detach().cpu(), output_landmark_offset.detach().cpu())
                        print("test r2_score  = ", score)
                        print("test landmark r2_score  = ", landmark_score)

                        # 总损失
                        test_loss = test_cls_loss + test_offset_loss + test_landmark_offset_loss

                        # 输出损失：loss-->gpu-->cup（变量）-->tensor-->array
                        print("i=", i, "loss:", test_loss.cpu().data.numpy(), "test_cls_loss:", test_cls_loss.cpu().data.numpy(),
                              "test_offset_loss",
                              test_offset_loss.cpu().data.numpy())

                        self.sumerWriter.add_scalars("loss",
                                                     {"train cls_loss": train_cls_loss,
                                                      "train offset_loss": train_offset_loss,
                                                      "test_cls_loss": test_cls_loss,
                                                      "test_offset_loss": test_offset_loss,
                                                      "test_landmark_offset_loss": test_landmark_offset_loss},
                                                     num)
                        num += 1
                        break

# 备注：
# [1] num_workers:表示有多少线程在工作;
#[2] lt:小于；gt:大于；eq:等于；le：小于等于；ge:大于等于
#[3] loss.cpu().data.numpy():把损失从从“cuda”里放到“cpu”,在根据data属性转成numpy数据