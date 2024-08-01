import cv2
from matplotlib import pyplot as plt

class ImageProcessing:
    def threshold():
        img = cv2.imread("./image_preprocessing/starbucks.jpg")

        # Implement Grayscale
        grayScaleImage = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        cv2.imshow('Gray',grayScaleImage)
        cv2.waitKey(0)

        _, inv_bin_thresh = cv2.threshold(grayScaleImage,100,255,cv2.THRESH_BINARY_INV)
        _, bin_thresh = cv2.threshold(grayScaleImage,100,255,cv2.THRESH_BINARY)
        _, trunc_thresh = cv2.threshold(grayScaleImage,100,255,cv2.THRESH_TRUNC)
        _, tozero_thresh = cv2.threshold(grayScaleImage,100,255,cv2.THRESH_TOZERO)
        _, inv_tozero_thresh = cv2.threshold(grayScaleImage,100,255,cv2.THRESH_TOZERO_INV)
        _, otsu_thresh = cv2.threshold(grayScaleImage,100,255,cv2.THRESH_OTSU)

        result = [grayScaleImage, bin_thresh,inv_bin_thresh,trunc_thresh,tozero_thresh,inv_tozero_thresh,otsu_thresh]
        dest = ['Grayscale','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV','OTSU']
        plt.figure('Threshold',figsize=(8,8))
        for i, (curr_img, curr_desc) in enumerate(zip(result,dest)):
            plt.subplot(3,3,(i+1))
            plt.imshow(curr_img,'gray')
            plt.title(curr_desc)
            plt.xticks([])
            plt.yticks([])
        plt.show()
    
    def filtering():
        img = cv2.imread('./image_preprocessing/starbucks.jpg')
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        mean_blur = cv2.blur(img,(11,11))
        gaussian_blur = cv2.GaussianBlur(img,(11,11),5.0)
        bilateral_blur = cv2.medianBlur(img,5,150,150)
        median_blur = cv2.medianBlur(img,11)

        image_result = [img,mean_blur,gaussian_blur,median_blur,bilateral_blur]
        desc = ['Original','Mean','Gaussian','Median','Bilateral']
        plt.figure('Blur',figsize=(8,8))
        for i, (curr_img, curr_desc) in enumerate(zip(image_result,desc)):
            plt.subplot(3,3,(i+1))
            plt.imshow(curr_img)
            plt.title(curr_desc)
            plt.xticks([])
            plt.yticks([])
        plt.show()