# MTCNN的使用
# 流程：图像-->缩放-->P网(NMS和边界框回归)-->R网路(NMS和边界框回归)--->O网络(NMS和边界框回归)

import torch
from PIL import Image
from PIL import ImageDraw
import numpy as np
import utils
import nets
from torchvision import transforms
import time
import os

import cv2

import datetime

# 网络调参
# P网络:
p_cls = 0.7 #原为0.6
p_nms = 0.1 #原为0.5

# R网络：
r_cls = 0.3 #原为0.6
r_nms = 0.5 #原为0.5

# O网络：
o_cls = 0.97 #原为0.97
o_nms = 0.3 #原为0.7


# 侦测器
class Detector:
    # 初始化时加载三个网络的权重(训练好的)，cuda默认设为True
    def __init__(self, pnet_param="./net_param/pnet.pt", rnet_param="./net_param/rnet.pt", onet_param="./net_param/onet.pt",
                 isCuda=True):

        self.isCuda = isCuda

        self.pnet = nets.PNet() # 创建实例变量，实例化P网络
        self.rnet = nets.RNet()
        self.onet = nets.ONet()

        if self.isCuda:
            self.pnet.cuda() # 给P网络加速
            self.rnet.cuda()
            self.onet.cuda()

        self.pnet.load_state_dict(torch.load(pnet_param)) # 把训练好的权重加载到P网络中
        self.rnet.load_state_dict(torch.load(rnet_param))
        self.onet.load_state_dict(torch.load(onet_param))

        self.pnet.eval() # 训练网络里有BN（批归一化时），要调用eval方法，使用是不用BN，dropout方法
        self.rnet.eval()
        self.onet.eval()

        self.__image_transform = transforms.Compose([
            transforms.ToTensor()
        ]) # 图片数据类型转换

    def detect(self, image): # 检测图片
        # P网络检测-----1st
        start_time = time.time()               # 开始计时
        pnet_boxes = self.__pnet_detect(image) # 调用__pnet_detect函数（后面定义）

        if pnet_boxes.shape[0] == 0:           # 若P网络没有人脸时，避免数据出错，返回一个新数组
            return np.array([])
        end_time = time.time()                 # 计时结束
        t_pnet = end_time - start_time         # P网络所占用的时间差
        # return pnet_boxes                    # p网络检测出的框

        # R网络检测-------2nd
        start_time = time.time()
        rnet_boxes = self.__rnet_detect(image, pnet_boxes) # 传入原图，P网络的一些框，根据这些框在原图上抠图
        if rnet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        t_rnet = end_time - start_time
        # return rnet_boxes

        #O网络检测--------3rd
        start_time = time.time()
        onet_boxes = self.__onet_detect(image, rnet_boxes) # 把原图和R网络里的框传到O网络里去
        if onet_boxes.shape[0] == 0:                      # 若P网络没有人脸时，避免数据出错，返回一个新数组
            return np.array([])
        end_time = time.time()
        t_onet = end_time - start_time

        # 三网络检测的总时间
        t_sum = t_pnet + t_rnet + t_onet
        print("total:{0} pnet:{1} rnet:{2} onet:{3}".format(t_sum, t_pnet, t_rnet, t_onet))

        return onet_boxes

    # 创建P网检测函数
    def __pnet_detect(self, image): # ★p网络全部是卷积，与输入图片大小无关，可输出任意形状图片
        boxes = [] # 创建空列表,接收符合条件的建议框

        img = image
        w, h = img.size
        min_side_len = min(w, h) # 获取图片的最小边长

        scale = 1 # 初始缩放比例（为1时不缩放）:得到不同分辨率的图片
        while min_side_len > 12: # 直到缩放到小于等于12时停止
            img_data = self.__image_transform(img) # 将图片数组转成张量
            if self.isCuda:
                img_data = img_data.cuda() # 将图片tensor传到cuda里加速
            img_data.unsqueeze_(0) # 在“批次”上升维（测试时传的不止一张图片）
            # print("img_data:",img_data.shape) # [1, 3, 416, 500]：C=3,W=416,H=500

            _cls, _offest, _landmark_offset = self.pnet(img_data) # ★★返回多个置信度和偏移量
            # print(_cls)
            # print(_offest.shape)
            # exit()
            # print("_cls",_cls.shape)         # [1, 1, 203, 245]:NCWH：分组卷积的特征图的通道和尺寸★
            # print("_offest", _offest.shape) # [1, 4, 203, 245]:NCWH

            cls= _cls[0][0].cpu().data  # [203, 245]：分组卷积特征图的尺寸：W,H
            offest = _offest[0].cpu().data  #[4, 203, 245] # 分组卷积特征图的通道、尺寸:C,W,H
            idxs = torch.nonzero(torch.gt(cls, p_cls)) # ★置信度大于0.6的框索引；把P网络输出，看有没没框到的人脸，若没框到人脸，说明网络没训练好；或者置信度给高了、调低

            for idx in idxs: # 根据索引，依次添加符合条件的框；cls[idx[0], idx[1]]在置信度中取值：idx[0]行索引，idx[1]列索引
                boxes.append(self.__box(idx, offest, cls[idx[0], idx[1]], scale)) # ★调用框反算函数_box（把特征图上的框，反算到原图上去），把大于0.6的框留下来；

            scale *= 0.7 # 缩放图片:循环控制条件
            _w = int(w * scale) # 新的宽度
            _h = int(h * scale)
            img = img.resize((_w, _h)) # 根据缩放后的宽和高，对图片进行缩放
            min_side_len = min(_w, _h) # 重新获取最小宽高

        return utils.nms(np.array(boxes), p_nms) #返回框框，原阈值给p_nms=0.5（iou为0.5），尽可能保留IOU小于0.5的一些框下来，若网络训练的好，值可以给低些

    # 特征反算：将回归量还原到原图上去，根据特征图反算的到原图建议框
    def __box(self, start_index, offset, cls, scale, stride=2, side_len=12): # p网络池化步长为2
        '''
        :param start_index: shape(W*H, 2)在 W*H 上的 索引。 start_index[0]行索引=>表示 y1 , start_index[1]列索引=>表示 x1
        :param offset: 偏移量
        :param cls: 置信度
        :param scale: 缩放比例
        :param stride: p网络池化步长为 2
        :param side_len: p网络边长 12
        :return:
        '''

        _x1 = (start_index[1].float() * stride) / scale # (索引*步长)/缩放比例；★特征反算时“行索引，索引互换”，原为[0]
        _y1 = (start_index[0].float() * stride) / scale
        _x2 = (start_index[1].float() * stride + side_len -1) / scale # (索引*步长 + 边长 - 1)/缩放
        _y2 = (start_index[0].float() * stride + side_len -1) / scale

        ow = _x2 - _x1  # 人脸所在区域建议框的宽和高
        oh = _y2 - _y1

        _offset = offset[:, start_index[0], start_index[1]] # 根据idxs行索引与列索引，找到对应偏移量△δ:[x1,y1,x2,y2]

        # ★ offset这里是一个偏移比值
        x1 = _x1 + ow * _offset[0] # 根据偏移量算实际框的位置，x1=x1_+w*△δ；生样时为:△δ=x1-x1_/w
        y1 = _y1 + oh * _offset[1]
        x2 = _x2 + ow * _offset[2]
        y2 = _y2 + oh * _offset[3]

        return [x1, y1, x2, y2, cls]  # 正式框：返回4个坐标点和1个偏移量

    # 创建R网络检测函数
    def __rnet_detect(self, image, pnet_boxes):

        _img_dataset = [] # 创建空列表，存放抠图
        _pnet_boxes = utils.convert_to_square(pnet_boxes) # ★给p网络输出的框，找出中心点，沿着最大边长的两边扩充成“正方形”，再抠图
        for _box in _pnet_boxes: # ★遍历每个框，每个框返回框4个坐标点，抠图，放缩，数据类型转换，添加列表
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop((_x1, _y1, _x2, _y2)) # 根据4个坐标点抠图
            img = img.resize((24, 24)) # 放缩在固尺寸
            img_data = self.__image_transform(img) # 将图片数组转成张量
            _img_dataset.append(img_data)

        img_dataset =torch.stack(_img_dataset) # stack堆叠(默认在0轴)，此处相当数据类型转换，见例子2★
        if self.isCuda:
            img_dataset = img_dataset.cuda() # 给图片数据采用cuda加速

        _cls, _offset, _landmark_offset = self.rnet(img_dataset) # ★★将24*24的图片传入网络再进行一次筛选

        cls = _cls.cpu().data.numpy() # 将gpu上的数据放到cpu上去，在转成numpy数组
        offset = _offset.cpu().data.numpy()
        # print("r_cls:",cls.shape)  # (11, 1):P网络生成了11个框
        # print("r_offset:", offset.shape)  # (11, 4)

        boxes = [] #R 网络要留下来的框，存到boxes里
        idxs, _ = np.where(cls > r_cls) # 原置信度0.6是偏低的，时候很多框并没有用(可打印出来观察)，可以适当调高些；idxs置信度框大于0.6的索引；★返回idxs:0轴上索引[0,1]，_:1轴上索引[0,0]，共同决定元素位置，见例子3
        for idx in idxs: # 根据索引，遍历符合条件的框；1轴上的索引，恰为符合条件的置信度索引（0轴上索引此处用不到）
            _box = _pnet_boxes[idx]
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            ow = _x2 - _x1 # 基准框的宽
            oh = _y2 - _y1

            x1 = _x1 + ow * offset[idx][0] # 实际框的坐标点
            y1 = _y1 + oh * offset[idx][1]
            x2 = _x2 + ow * offset[idx][2]
            y2 = _y2 + oh * offset[idx][3]

            boxes.append([x1, y1, x2, y2, cls[idx][0]]) # 返回4个坐标点和置信度

        return utils.nms(np.array(boxes), r_nms) # 原r_nms为0.5（0.5要往小调），上面的0.6要往大调;小于0.5的框被保留下来

    # 创建O网络检测函数
    def __onet_detect(self, image, rnet_boxes):

        _img_dataset = [] # 创建列表，存放抠图r
        _rnet_boxes = utils.convert_to_square(rnet_boxes) # 给r网络输出的框，找出中心点，沿着最大边长的两边扩充成“正方形”
        for _box in _rnet_boxes: # 遍历R网络筛选出来的框，计算坐标，抠图，缩放，数据类型转换，添加列表，堆叠
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop((_x1, _y1, _x2, _y2)) # 根据坐标点“抠图”
            img = img.resize((48, 48))
            img_data = self.__image_transform(img) # 将抠出的图转成张量
            _img_dataset.append(img_data)

        img_dataset = torch.stack(_img_dataset) # 堆叠，此处相当数据格式转换，见例子2
        if self.isCuda:
            img_dataset = img_dataset.cuda()

        _cls, _offset, _landmark_offset = self.onet(img_dataset)
        cls = _cls.cpu().data.numpy()       # (1, 1)
        offset = _offset.cpu().data.numpy() # (1, 4)
        landmark_offset = _landmark_offset.cpu().data.numpy() # (1, 10)

        boxes = [] # 存放o网络的计算结果
        idxs, _ = np.where(cls > o_cls) # 原o_cls为0.97是偏低的，最后要达到标准置信度要达到0.99999，这里可以写成0.99998，这样的话出来就全是人脸;留下置信度大于0.97的框；★返回idxs:0轴上索引[0]，_:1轴上索引[0]，共同决定元素位置，见例子3
        for idx in idxs: # 根据索引，遍历符合条件的框；1轴上的索引，恰为符合条件的置信度索引（0轴上索引此处用不到）
            _box = _rnet_boxes[idx] # 以R网络做为基准框
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])
            _cent_w = int(_x1 + _x2)/2
            _cent_y = int(_y1 + _y2)/2

            ow = _x2 - _x1 # 框的基准宽，框是“方”的，ow=oh
            oh = _y2 - _y1

            side_len = ow

            x1 = _x1 + ow * offset[idx][0] # O网络最终生成的框的坐标；生样，偏移量△δ=x1-_x1/w*side_len
            y1 = _y1 + oh * offset[idx][1]
            x2 = _x2 + ow * offset[idx][2]
            y2 = _y2 + oh * offset[idx][3]

            xx1 = _cent_w + side_len * landmark_offset[idx][0]
            yy1 = _cent_y + side_len * landmark_offset[idx][1]
            xx2 = _cent_w + side_len * landmark_offset[idx][2]
            yy2 = _cent_y + side_len * landmark_offset[idx][3]
            xx3 = _cent_w + side_len * landmark_offset[idx][4]
            yy3 = _cent_y + side_len * landmark_offset[idx][5]
            xx4 = _cent_w + side_len * landmark_offset[idx][6]
            yy4 = _cent_y + side_len * landmark_offset[idx][7]
            xx5 = _cent_w + side_len * landmark_offset[idx][8]
            yy5 = _cent_y + side_len * landmark_offset[idx][9]

            boxes.append([x1, y1, x2, y2,
                          cls[idx][0],
                          xx1, yy1,
                          xx2, yy2,
                          xx3, yy3,
                          xx4, yy4,
                          xx5, yy5]) #返回4个坐标点  和  1个置信度  和  landmark

        return utils.nms(np.array(boxes), o_nms, isMin=True) # 用最小面积的IOU；原o_nms(IOU)为小于0.7的框被保留下来


if __name__ == '__main__':
    # 多张图片检测
    image_path = r"test_images"
    aaa = 0
    for i in os.listdir(image_path):
        detector = Detector()
        print(os.path.join(image_path,i))
        with Image.open(os.path.join(image_path,i)) as im: # 打开图片

            print("----------------------------")
            boxes = detector.detect(im)

            imDraw = ImageDraw.Draw(im)
            for box in boxes: # 多个框，没循环一次框一个人脸
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])

                xx1 = int(box[5])
                yy1 = int(box[6])
                xx2 = int(box[7])
                yy2 = int(box[8])
                xx3 = int(box[9])
                yy3 = int(box[10])
                xx4 = int(box[11])
                yy4 = int(box[12])
                xx5 = int(box[13])
                yy5 = int(box[14])

                imDraw.rectangle((x1, y1, x2, y2), width=1)
                # imDraw.ellipse((xx1,yy1), fill="red", width=2)
                # imDraw.ellipse((xx2, yy2), fill="red", width=2)
                # imDraw.ellipse((xx3, yy3), fill="red", width=2)
                # imDraw.ellipse((xx4, yy4), fill="red", width=2)
                # imDraw.ellipse((xx5, yy5), fill="red", width=2)

                imDraw.point((xx1,yy1), fill="red")
                imDraw.point((xx2, yy2), fill="red")
                imDraw.point((xx3, yy3), fill="red")
                imDraw.point((xx4, yy4), fill="red")
                imDraw.point((xx5, yy5), fill="red")

                imDraw.text((x1, y1 - 8), str(box[4])[:4])
            im.show()
            # im.save(f"D:/Desktop/{aaa}.jpg")
            aaa += 1
            # exit()

# 备注：以上提到的例子1、2、3见“notes/13-detect”
