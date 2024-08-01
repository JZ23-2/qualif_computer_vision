import cv2
import numpy as np
from matplotlib import pyplot as plt

class EdgeDetection:
    def edgeDetection():
        img = cv2.imread("./edge_detection/dog.jpg")

        grayScale = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(grayScale,cv2.CV_64F)
        lap_uint8 = np.uint8(np.absolute(lap))

        sx = cv2.Sobel(grayScale,cv2.CV_64F,1,0,5)
        sy = cv2.Sobel(grayScale,cv2.CV_64F,0,1,5)

        canny = cv2.Canny(grayScale,100,200)

        result = [grayScale,lap,lap_uint8,sx,sy,canny]
        desc = ["Original","Laplacian","Laplacian Uint8","Sobel X", "Sobel Y", "Canny"]
        plt.figure('Edge Detection',figsize=(8,8))
        for i, (curr_img, curr_desc) in enumerate(zip(result,desc)):
            plt.subplot(3,3,(i+1))
            plt.imshow(curr_img,'gray')
            plt.title(curr_desc)
            plt.xticks([])
            plt.yticks([])
        plt.show()
