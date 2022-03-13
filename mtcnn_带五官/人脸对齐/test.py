#coding=utf-8
import os,cv2,numpy
import logging

logging.basicConfig(
	level=logging.DEBUG,
	format='%(asctime)s %(levelname)s: %(message)s',
	datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

imgSize = [112, 96];

scale = 112/96

coord5point = [[30.2946, 51.6963],
               [65.5318, 51.6963],
               [48.0252, 71.7366],
               [33.5493, 92.3655],
               [62.7299, 92.3655]]

imgSize = [112, 112];
coord5point = [[30.2946, 51.6963 * scale],
               [65.5318, 51.6963 * scale],
               [48.0252, 71.7366 * scale],
               [33.5493, 92.3655 * scale],
               [62.7299, 92.3655 * scale]]

coord5point = [[30.2946 * scale, 51.6963],
               [65.5318 * scale, 51.6963],
               [48.0252 * scale, 71.7366],
               [33.5493 * scale, 92.3655],
               [62.7299 * scale, 92.3655]]

# 000010.jpg    142 176 301 349      178 215 260 204 228 259 196 296 271 283

face_landmarks = [[178, 215],
                  [260, 204],
                  [228, 259],
                  [196, 296],
                  [271, 283]]

def transformation_from_points(points1, points2):
    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)
    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2
    U, S, Vt = numpy.linalg.svd(points1.T * points2)
    R = (U * Vt).T
    return numpy.vstack([numpy.hstack(((s2 / s1) * R,c2.T - (s2 / s1) * R * c1.T)),numpy.matrix([0., 0., 1.])])

def warp_im(img_im, orgi_landmarks,tar_landmarks):
    pts1 = numpy.float64(numpy.matrix([[point[0], point[1]] for point in orgi_landmarks]))
    pts2 = numpy.float64(numpy.matrix([[point[0], point[1]] for point in tar_landmarks]))
    M = transformation_from_points(pts1, pts2)
    dst = cv2.warpAffine(img_im, M[:2], (img_im.shape[1], img_im.shape[0])) #仿射变换
    return dst

def main():
    pic_path = r'D:\Desktop\Celeba2\img\000010.jpg'
    img_im = cv2.imread(pic_path)

    for point in face_landmarks:
        cv2.circle(img_im, point, radius=2, color=(0,0,255), thickness=2)
    cv2.rectangle(img_im, (142, 176), (301,349), color=(0,0,255), thickness=2)
    cv2.imshow('affine_img_im', img_im)

    dst = warp_im(img_im, face_landmarks, coord5point)

    cv2.rectangle(dst, (142, 176), (301, 349), color=(0, 0, 255), thickness=2)
    cv2.imshow('affine', dst)

    crop_im = dst[0:imgSize[0], 0:imgSize[1]]


    cv2.imshow('affine_crop_im', crop_im)

if __name__=='__main__':
    main()
    cv2.waitKey()
    pass
