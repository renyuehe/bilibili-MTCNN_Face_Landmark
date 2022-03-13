import cv2
import os

if __name__ == '__main__':

    file = open(r"D:\Desktop\celeba\label\label_box_landmark.txt")
    ret = file.readline()
    ret = file.readline()
    lt = ret.split()

    xx1 = int(lt[1])
    yy1 = int(lt[2])
    xx2 = int(lt[3])
    yy2 = int(lt[4])

    x1 = int(lt[5])
    y1 = int(lt[6])
    x2 = int(lt[7])
    y2 = int(lt[8])
    x3 = int(lt[9])
    y3 = int(lt[10])
    x4 = int(lt[11])
    y4 = int(lt[12])
    x5 = int(lt[13])
    y5 = int(lt[14])

    img = cv2.imread(r"D:\Desktop\data\CelebA\Img\img_celeba.7z\img_celeba\000002.jpg")

    # 要画的点的坐标
    points_list = [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5)]

    for point in points_list:
        print(point)
        cv2.circle(img, center=point, radius=1, color=(0, 0, 255), thickness=2)

    cv2.rectangle(img, pt1=(xx1, yy1), pt2=(xx2, yy2), color=(255,0,0), thickness=2)

    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()