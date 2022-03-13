import numpy
import numpy as np
import torch

#重叠率
def iou(box, boxes, isMin = False): #1st框，一堆框，inMin(IOU有两种：一个除以最小值，一个除以并集)
    #计算面积：[x1,y1,x2,y3]
    box_area = (box[2] - box[0]) * (box[3] - box[1]) #原始框的面积
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])  #数组代替循环

    #找交集：
    xx1 = np.maximum(box[0], boxes[:, 0]) #横坐标，左上角最大值
    yy1 = np.maximum(box[1], boxes[:, 1]) #纵坐标，左上角最大值
    xx2 = np.minimum(box[2], boxes[:, 2]) #横坐标，右下角最小值
    yy2 = np.minimum(box[3], boxes[:, 3]) #纵坐标，右小角最小值

    # 判断是否有交集
    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)

    #交集的面积
    inter = w * h  #对应位置元素相乘

    if isMin: #若果为False
        ovr = np.true_divide(inter, np.minimum(box_area, area)) #最小面积的IOU：O网络用
    else:
        ovr = np.true_divide(inter, (box_area + area - inter))  #并集的IOU：P和R网络用；交集/并集

    return ovr

#非极大值抑制
#思路：首先根据对置信度进行排序，找出最大值框与每个框做IOU比较，再讲保留下来的框再进行循环比较，知道符合条件，保留其框
def nms(boxes, thresh=0.3, isMin = False):
    #框的长度为0时(防止程序有缺陷报错)
    if boxes.shape[0] == 0:
        return np.array([])

    #框的长度不为0时
    #根据置信度排序：[x1,y1,x2,y2,C]
    _boxes = boxes[(-boxes[:, 4]).argsort()] # #根据置信度“由大到小”，默认有小到大（加符号可反向排序）

    #创建空列表，存放保留剩余的框
    r_boxes = []
    # 用1st个框，与其余的框进行比较，当长度小于等于1时停止（比len(_boxes)-1次）
    while _boxes.shape[0] > 1: #shape[0]等价于shape(0),代表0轴上框的个数（维数）
        #取出第1个框
        a_box = _boxes[0]
        #取出剩余的框
        b_boxes = _boxes[1:]

        #将1st个框加入列表
        r_boxes.append(a_box) ##每循环一次往，添加一个框

        #比较IOU，将符合阈值条件的的框保留下来
        index = np.where(iou(a_box, b_boxes,isMin) < thresh) #将阈值小于0.3的建议框保留下来，返回保留框的索引

        _boxes = b_boxes[index] #循环控制条件；取出阈值小于0.3的建议框

    if _boxes.shape[0] > 0: ##最后一次，结果只用1st个符合或只有一个符合，若框的个数大于1；★此处_boxes调用的是whilex循环里的，此判断条件放在循环里和外都可以（只有在函数类外才可产生局部作用于）
        r_boxes.append(_boxes[0]) #将此框添加到列表中
    #stack组装为矩阵：:将列表中的数据在0轴上堆叠（行方向）
    return np.stack(r_boxes)

#扩充：找到中心点，及最大边长，沿着最大边长的两边扩充
def convert_to_square(bbox): #将长方向框，补齐转成正方形框
    square_bbox = bbox.copy()
    if bbox.shape[0] == 0:
        return np.array([])
    h = bbox[:, 3] - bbox[:, 1] #框高
    w = bbox[:, 2] - bbox[:, 0] #框宽
    max_side = np.maximum(h, w) #返回最大边长
    square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_side * 0.5
    square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - max_side * 0.5
    square_bbox[:, 2] = square_bbox[:, 0] + max_side #加最大边长，加最大边长，决定了正方形框
    square_bbox[:, 3] = square_bbox[:, 1] + max_side
    return square_bbox


if __name__ == '__main__':
# ★★★生成样本是不注释会有影响
# #注释：
# #例子1--max与maxium的区别
# import numpy as np
#
# a = [-2,-1,0,1,2]
# print(np.max(a)) #接收一个参数，返回最大值
# print(np.maximum(0,a)) #接收两个参数，X与Y逐个比较取其最大值：若比0小返回0，若比0大返回较大值
#
# #例子2--true_divide等价于divide,在python3中
#
# #例子3----argsort根据索引排序
# b = np.array([[5,3],[3,2],[1,6]])
# index = -b[:,1].argsort() #列出b[:,1]对应元素的负值索引
# print(b[index]) #根据b[:,1]对应元素的索引由大到小排序，不加符号由小到大排序；[[1 6]，[5 3],[3 2]]
#
# #例子4----a[np.where(a<3)]:返回符合条件的值
# c = np.array([5,3,2,1,6])
#
# print(c<3) #[False False  True  True False]
# index = np.where(c<3) #返回符合条件的索引
# print(index) #[2, 3]
# print(a[index]) #返回符合条件的值：[2 1]
#
# #例子5----np.stack的用方法
# a = np.array([1,2])
# b = np.array([3,4])
# c = np.array([5,6])
#
# list = []
#
# list.append(a)
# list.append(b)
# list.append(c)
#
# print(list) # [array([1, 2]), array([3, 4]), array([5, 6])]
#
# d = np.stack(list)
# print(d) #[[1 2][3 4][5 6]]


    # a = np.array([1, 1, 10, 10, 40])
    # b = np.array([[ 9,  8, 13, 20, 15]])
    #
    # print(iou(a,b))
    #
    # aa = torch.tensor([1, 1, 10, 10, 40])
    # bb = torch.tensor([[ 9,  8, 13, 20, 15]])
    #
    # print(iou_torch(aa, bb))

    bs = np.array([[1, 1, 10, 10, 40],
                          [1, 1, 9, 9, 10],
                          [9, 8, 13, 20, 15],
                          [6, 11, 18, 17, 13]])
    print(nms(bs, 0.1))


    boxes = torch.tensor([[1, 1, 10, 10, 40],
                          [1, 1, 9, 9, 10],
                          [9, 8, 13, 20, 15],
                          [6, 11, 18, 17, 13]])
    print(nms_torch(boxes, 0.1))
