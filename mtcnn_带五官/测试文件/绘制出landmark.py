from torchvision import datasets
from torchvision import transforms

import os
from functools import reduce
import cv2

if __name__ == '__main__':
    file = open(r"D:\Desktop\data\CelebA\Anno\list_landmarks_align_celeba.txt", 'r')
    ret = file.readline()
    ret = file.readline()
    ret = file.readline()
    ret = file.readline()
    lt = ret.split()
    print(lt)

    x1 = int(lt[1])
    y1 = int(lt[2])
    x2 = int(lt[3])
    y2 = int(lt[4])
    x3 = int(lt[5])
    y3 = int(lt[6])
    x4 = int(lt[7])
    y4 = int(lt[8])
    x5 = int(lt[9])
    y5 = int(lt[10])


    img = cv2.imread(r"D:\Desktop\data\CelebA\Img\img_align_celeba_png.7z\img_align_celeba_png\000002.png")
    print(img.shape)

    # 要画的点的坐标
    points_list = [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5)]

    for point in points_list:
        print(point)
        cv2.circle(img, center=point, radius=1, color=(0, 0, 255), thickness=2)

    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ...
