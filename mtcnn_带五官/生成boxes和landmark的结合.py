from torchvision import datasets
from torchvision import transforms

import os
from functools import reduce
import cv2

if __name__ == '__main__':
    file = open(r"D:\Desktop\data\CelebA\Anno\list_landmarks_celeba.txt", 'r')  # cebelA landmark 文件
    file.readline()
    file.readline()
    file2 = open(r"D:\Desktop\data\CelebA\Anno\label.txt", 'r') # cebelA boxes 文件
    file3 = open(r"D:\Desktop\celeba\label\label_box_landmark.txt", "a") #追加模式打开

    flag = True
    complete_list = []
    complete_str = ''
    i = 0
    while True:
        ret = file.readline()
        ret2 = file2.readline()

        lt = ret.split()
        lt2 = ret2.split()

        try:
            if lt[0] == lt2[0]: # 文件名相等
                print(lt[0], lt2[0])

                complete_list.clear()
                complete_list = [lt[0]] + lt2[1:] + lt[1:]
                complete_str = " ".join(str(i) for i in complete_list) + "\n"

                file3.write(complete_str)

                print(complete_str)

            else:# 文件名不相等情况
                lt_int = int((lt[0].split("."))[0])
                lt2_int = int((lt2[0].split("."))[0])
                print(lt_int, lt2_int)
                if lt_int > lt2_int:
                    file2.readline()
                else:
                    file.readline()

        except ValueError as e:
            print(e)
            continue
        except Exception as e:
            print(e)
            break

    file.close()
    file2.close()
    file3.close()
