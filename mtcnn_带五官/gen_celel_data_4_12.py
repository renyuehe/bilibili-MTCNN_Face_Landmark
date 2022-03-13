import os

import cv2
from PIL import Image
import numpy as np
import utils
import traceback
# from tool.parse_xml import get_label

# 正式生成样本
# 原始数据样本、标签路径

negative_time = 25
negative_other_time = 25
positive_time = 25
part_time = 5

anno_dir = r"D:\Desktop\Celeba2\label"
img_dir = r"D:\Desktop\Celeba2\img"

# 样本保存路径
save_path = r"D:\Desktop\Celeba2\celeba_3"

# 生成不同尺寸的人脸样本，包括人脸（正样本）、非人脸（负样本）、部分人脸
for face_size in [12]:
    print("gen %i image" % face_size)    # %i:十进制数占位符
    # “样本图片”存储路径--image
    positive_image_dir = os.path.join(save_path, str(face_size), "positive") # 三级文件路径
    negative_image_dir = os.path.join(save_path, str(face_size), "negative")
    part_image_dir = os.path.join(save_path, str(face_size), "part")

    for dir_path in [positive_image_dir, negative_image_dir, part_image_dir]:
        if not os.path.exists(dir_path):  # 如果文件不存在则创建文件路径
            os.makedirs(dir_path)


    # “样本标签”存储路径--text
    positive_anno_filename = os.path.join(save_path, str(face_size), "positive.txt") # 创建正样本txt文件
    negative_anno_filename = os.path.join(save_path, str(face_size), "negative.txt")
    part_anno_filename = os.path.join(save_path, str(face_size), "part.txt")


    # 计数初始值:给文件命名
    negative_list = os.listdir(negative_image_dir)
    if negative_list.__len__() == 0:
        negative_count = 0
    else:
        negative_list = [int(i.split(".")[0]) for i in negative_list]
        negative_count = max(negative_list) + 1


    positive_list = os.listdir(positive_image_dir)
    if negative_list.__len__() == 0:
        positive_count = 0
    else:
        positive_list = [int(i.split(".")[0]) for i in positive_list]
        positive_count = max(positive_list) + 1


    part_list = os.listdir(part_image_dir)
    if negative_list.__len__() == 0:
        part_count = 0
    else:
        part_list = [int(i.split(".")[0]) for i in part_list]
        part_count = max(part_list) + 1
    # 凡是文件操作，最好try一下，防止程序出错奔溃
    try:
        positive_anno_file = open(positive_anno_filename, "a") # 以写入的模式打开txt文档
        negative_anno_file = open(negative_anno_filename, "a")
        part_anno_file = open(part_anno_filename, "a")

        # 获取所有的 xml 文件, 以及对应的路径
        label_list = open(f"{anno_dir}\label_box_landmark.txt")
        for i, line in enumerate(label_list):
            try:
                # strs = label[3]
                # image_file = label[0][0]
                strs = line.strip().split()
                if strs.__len__() == 0:
                    continue
                image_file = strs[0].strip() # 文件名

                image_path_file = os.path.join(img_dir, image_file)  # 创建文件绝对路径

                with Image.open(image_path_file) as img:  # 打开图片文件
                    img_w, img_h = img.size
                    x1 = float(strs[1].strip()) # 取2nd个值去除两边的空格，再转车float型
                    y1 = float(strs[2].strip())
                    x2 = float(strs[3].strip())
                    y2 = float(strs[4].strip())

                    xx1 = float(strs[5].strip())
                    yy1 = float(strs[6].strip())
                    xx2 = float(strs[7].strip())
                    yy2 = float(strs[8].strip())
                    xx3 = float(strs[9].strip())
                    yy3 = float(strs[10].strip())
                    xx4 = float(strs[11].strip())
                    yy4 = float(strs[12].strip())
                    xx5 = float(strs[13].strip())
                    yy5 = float(strs[14].strip())

                    # 过滤字段，去除不符合条件的坐标
                    w = x2 - x1
                    h = y2 - y1

                    # 过滤不合格数据
                    if max(w, h) < 40 or x1 < 0 or y1 < 0 or w < 0 or h < 0:
                        continue

                    # 标注不太标准：给人脸框与适当的偏移★
                    boxes = [[x1, y1, x2, y2]]   #左上角和右下角四个坐标点；二维的框有批次概念

                    # 计算出人脸中心点位置：框的中心位置
                    cx = x1 + w / 2
                    cy = y1 + h / 2

                    # 五官相对于中心点的偏移
                    xx1_off = xx1 - cx
                    yy1_off = yy1 - cy
                    xx2_off = xx2 - cx
                    yy2_off = yy2 - cy
                    xx3_off = xx3 - cx
                    yy3_off = yy3 - cy
                    xx4_off = xx4 - cx
                    yy4_off = yy4 - cy
                    xx5_off = xx5 - cx
                    yy5_off = yy5 - cy

                    # 备份
                    _boxes = np.array(boxes)

                    # 增加正样本
                    for i in range(positive_time):  # 数量一般和前面保持一样
                        # 让人脸中心点有少许的偏移
                        w_ = np.random.randint(-w * 0.3, w * 0.3)  # 框的横向偏移范围：向左、向右移动了20%
                        h_ = np.random.randint(-h * 0.3, h * 0.3)

                        # 建议框的中心点
                        cx_ = cx + w_
                        cy_ = cy + h_

                        # 让人脸形成正方形（12*12，24*24,48*48），并且让坐标也有少许的偏离
                        side_len = np.random.randint(int(min(w, h) * 0.7), np.ceil(
                            1.25 * max(w, h)))  # 边长偏移的随机数的范围；ceil大于等于该值的最小整数（向上取整）;原0.8

                        # 建议框相比原框的缩放系数
                        scale_h = side_len/h
                        scale_w = side_len/w

                        side_len = min((side_len, min((img_h, img_w))))  # 确保 边长小于原始图片宽和长

                        # 左上角处理
                        x1_ = np.maximum(cx_ - side_len / 2,  0)  # 求左上角坐标, 保证不会小于0
                        y1_ = np.maximum(cy_ - side_len / 2,  0)  # 求左上角坐标, 保证不会小于0

                        x2_ = x1_ + side_len
                        y2_ = y1_ + side_len

                        # 右下角处理
                        if x2_ > img_w:  # 若 右下坐标超出图片处理
                            x2_ = x2_ if x2_ < img_w else img_w
                            x1_ = x2_ - side_len
                        if y2_ > img_h:
                            y2_ = y2_ if y2_ < img_h else img_h
                            y1_ = y2_ - side_len

                        # # 特征点的建议框位置（不带缩放系数）
                        # xx1_ = xx1_off + cx_
                        # yy1_ = xx1_off + cy_
                        # xx2_ = xx2_off + cx_
                        # yy2_ = xx2_off + cy_
                        # xx3_ = xx3_off + cx_
                        # yy3_ = xx3_off + cy_
                        # xx4_ = xx4_off + cx_
                        # yy4_ = xx4_off + cy_
                        # xx5_ = xx5_off + cx_
                        # yy5_ = xx5_off + cy_

                        # 特征点的建议框位置（带缩放系数）
                        xx1_ = (xx1_off + cx_) * scale_w
                        yy1_ = (xx1_off + cy_) * scale_h
                        xx2_ = (xx2_off + cx_) * scale_w
                        yy2_ = (xx2_off + cy_) * scale_h
                        xx3_ = (xx3_off + cx_) * scale_w
                        yy3_ = (xx3_off + cy_) * scale_h
                        xx4_ = (xx4_off + cx_) * scale_w
                        yy4_ = (xx4_off + cy_) * scale_h
                        xx5_ = (xx5_off + cx_) * scale_w
                        yy5_ = (xx5_off + cy_) * scale_h

                        crop_box = np.array([x1_, y1_, x2_, y2_])  # 偏移后的新框

                        # 计算坐标的偏移值
                        offset_x1 = (x1 - x1_) / side_len # 偏移量△δ=(x1-x1_)/side_len;新框的宽度;★????还要梳理，可打印出来观察
                        offset_y1 = (y1 - y1_) / side_len
                        offset_x2 = (x2 - x2_) / side_len
                        offset_y2 = (y2 - y2_) / side_len

                        # 计算特征点偏移率
                        offset_xx1 = (xx1 - xx1_) / side_len # 偏移量△δ=(x1-x1_)/side_len;新框的宽度;★????还要梳理，可打印出来观察
                        offset_yy1 = (yy1 - yy1_) / side_len
                        offset_xx2 = (xx2 - xx2_) / side_len
                        offset_yy2 = (yy2 - yy2_) / side_len
                        offset_xx3 = (xx3 - xx3_) / side_len
                        offset_yy3 = (yy3 - yy3_) / side_len
                        offset_xx4 = (xx4 - xx4_) / side_len
                        offset_yy4 = (yy4 - yy4_) / side_len
                        offset_xx5 = (xx5 - xx5_) / side_len
                        offset_yy5 = (yy5 - yy5_) / side_len


                        if utils.iou(crop_box, _boxes) > 0.7:  # 在加IOU进行判断：保留小于0.3的那一部分；原为0.3
                            face_crop = img.crop(crop_box)  # 抠图
                            face_resize = face_crop.resize((face_size, face_size), Image.ANTIALIAS)  # ANTIALIAS：平滑,抗锯齿

                            # 0：置信度、1：位置、 2-5：偏移量、5-15：五官？？
                            positive_anno_file.write(
                                "positive/{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n".format(
                                    positive_count, 1,
                                    offset_x1,  offset_y1,
                                    offset_x2,  offset_y2,
                                    offset_xx1, offset_yy1,
                                    offset_xx2, offset_yy2,
                                    offset_xx3, offset_yy3,
                                    offset_xx4, offset_yy4,
                                    offset_xx5, offset_yy5))
                            positive_anno_file.flush()
                            face_resize.save(os.path.join(positive_image_dir, "{0}.jpg".format(positive_count)))
                            positive_count += 1

                    # 生成部分样本
                    for i in range(part_time):
                        # 让人脸中心点有少许的偏移
                        w_ = np.random.randint(-w * 0.7, w * 0.7)  # 框的横向偏移范围：向左、向右移动了20%
                        h_ = np.random.randint(-h * 0.7, h * 0.7)
                        cx_ = cx + w_
                        cy_ = cy + h_


                        # 让人脸形成正方形（12*12，24*24,48*48），并且让坐标也有少许的偏离
                        side_len = np.random.randint(int(min(w, h) * 0.7), np.ceil(
                            1.25 * max(w, h)))  # 边长偏移的随机数的范围；ceil大于等于该值的最小整数（向上取整）;原0.8

                        # 建议框相比原框的缩放系数
                        scale_h = side_len/h
                        scale_w = side_len/w

                        side_len = min((side_len, min((img_h, img_w))))  # 确保 边长小于原始图片宽和长

                        x1_ = np.maximum(cx_ - side_len / 2, 0)  # 坐标点随机偏移
                        y1_ = np.maximum(cy_ - side_len / 2, 0)

                        x2_ = x1_ + side_len
                        y2_ = y1_ + side_len

                        # 右下角处理
                        if x2_ > img_w:  # 若 右下坐标超出图片处理
                            x2_ = x2_ if x2_ < img_w else img_w
                            x1_ = x2_ - side_len
                        if y2_ > img_h:
                            y2_ = y2_ if y2_ < img_h else img_h
                            y1_ = y2_ - side_len

                        # # 特征点的建议框位置（不带缩放系数）
                        # xx1_ = xx1_off + cx_
                        # yy1_ = xx1_off + cy_
                        # xx2_ = xx2_off + cx_
                        # yy2_ = xx2_off + cy_
                        # xx3_ = xx3_off + cx_
                        # yy3_ = xx3_off + cy_
                        # xx4_ = xx4_off + cx_
                        # yy4_ = xx4_off + cy_
                        # xx5_ = xx5_off + cx_
                        # yy5_ = xx5_off + cy_

                        # 特征点的建议框位置（带缩放系数）
                        xx1_ = (xx1_off + cx_) * scale_w
                        yy1_ = (xx1_off + cy_) * scale_h
                        xx2_ = (xx2_off + cx_) * scale_w
                        yy2_ = (xx2_off + cy_) * scale_h
                        xx3_ = (xx3_off + cx_) * scale_w
                        yy3_ = (xx3_off + cy_) * scale_h
                        xx4_ = (xx4_off + cx_) * scale_w
                        yy4_ = (xx4_off + cy_) * scale_h
                        xx5_ = (xx5_off + cx_) * scale_w
                        yy5_ = (xx5_off + cy_) * scale_h

                        crop_box = np.array([x1_, y1_, x2_, y2_])  # 偏移后的新框

                        # 计算坐标的偏移值
                        offset_x1 = (x1 - x1_) / side_len # 偏移量△δ=(x1-x1_)/side_len;新框的宽度;★????还要梳理，可打印出来观察
                        offset_y1 = (y1 - y1_) / side_len
                        offset_x2 = (x2 - x2_) / side_len
                        offset_y2 = (y2 - y2_) / side_len

                        # 计算特征点偏移率
                        offset_xx1 = (xx1 - xx1_) / side_len # 偏移量△δ=(x1-x1_)/side_len;新框的宽度;★????还要梳理，可打印出来观察
                        offset_yy1 = (yy1 - yy1_) / side_len
                        offset_xx2 = (xx2 - xx2_) / side_len
                        offset_yy2 = (yy2 - yy2_) / side_len
                        offset_xx3 = (xx3 - xx3_) / side_len
                        offset_yy3 = (yy3 - yy3_) / side_len
                        offset_xx4 = (xx4 - xx4_) / side_len
                        offset_yy4 = (yy4 - yy4_) / side_len
                        offset_xx5 = (xx5 - xx5_) / side_len
                        offset_yy5 = (yy5 - yy5_) / side_len

                        iou = utils.iou(crop_box, _boxes)
                        if iou> 0.25 and iou < 0.7:   # 在加IOU进行判断：保留小于0.3的那一部分；原为0.3
                            face_crop = img.crop(crop_box)  # 抠图
                            face_resize = face_crop.resize((face_size, face_size), Image.ANTIALIAS) #ANTIALIAS：平滑,抗锯齿

                            part_anno_file.write("part/{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n".format(
                                part_count, 2,
                                offset_x1, offset_y1,
                                offset_x2, offset_y2,
                                offset_xx1, offset_yy1,
                                offset_xx2, offset_yy2,
                                offset_xx3, offset_yy3,
                                offset_xx4, offset_yy4,
                                offset_xx5, offset_yy5
                            ))
                            part_anno_file.flush()
                            face_resize.save(os.path.join(part_image_dir, "{0}.jpg".format(part_count)))
                            part_count += 1
                        pass

                    # 生成负样本
                    for i in range(negative_time):
                        # 让人脸中心点有少许的偏移
                        w_ = np.random.randint(-w * 1, w * 1)  # 框的横向偏移范围：向左、向右移动了20%
                        h_ = np.random.randint(-h * 1, h * 1)
                        cx_ = cx + w_
                        cy_ = cy + h_

                        # 让人脸形成正方形（12*12，24*24,48*48），并且让坐标也有少许的偏离
                        side_len = np.random.randint(int(min(w, h) * 0.5),
                                                     np.ceil(1.5 * max(w, h)))  # 边长偏移的随机数的范围；ceil大于等于该值的最小整数（向上取整）;原0.8

                        side_len = min((side_len, min((img_h, img_w))))  # 确保 边长小于原始图片宽和长


                        x1_ = np.maximum(cx_ - side_len / 2, 0)  # 坐标点随机偏移
                        y1_ = np.maximum(cy_ - side_len / 2, 0)

                        x2_ = x1_ + side_len
                        y2_ = y1_ + side_len

                        # 右下角处理
                        if x2_ > img_w:  # 若 右下坐标超出图片处理
                            x2_ = x2_ if x2_ < img_w else img_w
                            x1_ = x2_ - side_len
                        if y2_ > img_h:
                            y2_ = y2_ if y2_ < img_h else img_h
                            y1_ = y2_ - side_len

                        crop_box = np.array([x1_, y1_, x2_, y2_])  # 偏移后的新框
                        if utils.iou(crop_box, _boxes) < 0.02:  # 在加IOU进行判断：保留小于0.3的那一部分；原为0.3
                            face_crop = img.crop(crop_box)  # 抠图
                            face_resize = face_crop.resize((face_size, face_size), Image.ANTIALIAS)  # ANTIALIAS：平滑,抗锯齿

                            negative_anno_file.write(
                                "negative/{0}.jpg {1} 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n".format(negative_count, 0))
                            negative_anno_file.flush()
                            face_resize.save(os.path.join(negative_image_dir, "{0}.jpg".format(negative_count)))
                            negative_count += 1

                    # 补充场外负样本
                    for i in range(negative_other_time):
                        if(face_size > min(img_w, img_h)/2):
                            continue
                        side_len = np.random.randint(face_size, min(img_w, img_h) / 2)
                        x_ = np.random.randint(0, img_w - side_len)
                        y_ = np.random.randint(0, img_h - side_len)
                        crop_box = np.array([x_, y_, x_ + side_len, y_ + side_len])

                        if utils.iou(crop_box, _boxes) < 0.25:   # 在加IOU进行判断：保留小于0.3的那一部分；原为0.3
                            face_crop = img.crop(crop_box)  # 抠图
                            face_resize = face_crop.resize((face_size, face_size), Image.ANTIALIAS) #ANTIALIAS：平滑,抗锯齿

                            negative_anno_file.write("negative/{0}.jpg {1} 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n".format(negative_count, 0))
                            negative_anno_file.flush()
                            face_resize.save(os.path.join(negative_image_dir, "{0}.jpg".format(negative_count)))
                            negative_count += 1

            except Exception as e:
                traceback.print_exc()  # 如果出现异常，把异常打印出来

    #关闭写入文件
    finally:
        positive_anno_file.close() #关闭正样本txt件
        negative_anno_file.close()
        part_anno_file.close()
