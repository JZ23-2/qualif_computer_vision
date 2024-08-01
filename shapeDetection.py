import cv2
from matplotlib import pyplot as plt

class ShapeDetection:
    def shapeDetection():
        object_img= cv2.imread('./shape_detection/4-starBall.jpg')
        scene_img= cv2.imread('./shape_detection/dragonBall.jpg')

        surf = cv2.SIFT.create()
        kp_object, des_object = surf.detectAndCompute(object_img,None)
        kp_scene, des_scene = surf.detectAndCompute(scene_img,None)

        des_object = des_object.astype('f')
        des_scene = des_scene.astype('f')

        FLANN_INDEX = 0
        index_params = dict(algorithm=FLANN_INDEX)
        search_params = dict(checks=100)

        flan = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flan.knnMatch(des_object,des_scene,2)

        matchesMask = []
        for _ in range (0,len(matches)):
            matchesMask.append([0,0])
        
        total_match = 0
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                matchesMask[i] = [1,0]
                total_match += 1
        
        img_res = cv2.drawMatchesKnn(object_img,kp_object,scene_img,kp_scene,matches,None,matchColor=[0,255,0],singlePointColor=[255,0,0], matchesMask=matchesMask)

        plt.figure('Shape Detection')
        plt.imshow(img_res)
        plt.show()