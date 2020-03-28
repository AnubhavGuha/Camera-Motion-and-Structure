# coding in UTF-8

import numpy as np
import cv2
import matplotlib.pyplot as plt
import operator


img_l = cv2.imread('../../images/task_7/left_0.png')
img_r = cv2.imread('../../images/task_7/left_7.png')

images = []
images.append(img_l)
images.append(img_r)
gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

fs_l = cv2.FileStorage("../../parameters/left_camera_intrinsics.xml", cv2.FILE_STORAGE_READ)
cameraMatrix_l = fs_l.getNode("camera_intrinsic")
#print(cameraMatrix_l.mat())
distMatrix_l = fs_l.getNode("distort_coefficients")

fs_r = cv2.FileStorage("../../parameters/right_camera_intrinsics.xml", cv2.FILE_STORAGE_READ)
cameraMatrix_r = fs_r.getNode("camera_intrinsic")
distMatrix_r = fs_r.getNode("distort_coefficients")


# Undistort left image
h, w = gray_l.shape[:2]
#newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix_l.mat(), distMatrix_l.mat(), (w, h), 1, (w, h))
mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix_l.mat(), distMatrix_l.mat(), None, cameraMatrix_l.mat(), (w, h), 5)
gray_l = cv2.remap(img_l, mapx, mapy, cv2.INTER_LINEAR)
#x, y, w, h = roi
#gray_l = dst[y:y + h, x:x + w]
#cv2.imwrite(r'../../output/task_7/l_distort.png', dst)

# Undistort right image
h, w = gray_r.shape[:2]
#newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix_r.mat(), distMatrix_r.mat(), (w, h), 1, (w, h))
mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix_r.mat(), distMatrix_r.mat(), None, cameraMatrix_r.mat(), (w, h), 5)
gray_r = cv2.remap(img_l, mapx, mapy, cv2.INTER_LINEAR)
#x, y, w, h = roi
#gray_r = dst[y:y + h, x:x + w]
#cv2.imwrite(r'../../output/task_7/r_distort.png', dst)


#ORB
orb = cv2.ORB_create()
kp_l = orb.detect(gray_l, None)
#img2_l = cv2.drawKeypoints(gray_l, kp_l, None, color=(0,255,0), flags=0)
#plt.imsave("../../output/task_7/l_key_points.png", img2_l)

kp_r = orb.detect(gray_r, None)
#img2_r = cv2.drawKeypoints(gray_r, kp_r, None, color=(0,255,0), flags=0)
#plt.imsave("../../output/task_7/r_key_points.png", img2_r)

# left keypoints
keypoint_list_l = []
for i, keypoint in enumerate(kp_l):
    #print("Keypoint:", i, keypoint)
    keypoint_list_l.append(keypoint)

# sort by response
cmpfun = operator.attrgetter('response')
keypoint_list_l.sort(key=cmpfun, reverse=True)

# find minimum
distance = []
radius_l = []
keypoint_i = 0
for keypoint in keypoint_list_l:
    # print("Keypoint:", keypoint.response)
    distance.append([])
    if keypoint_i == 0:
        distance[0].append(1)
    for index in range(keypoint_i):
        distance[keypoint_i].append(np.linalg.norm(np.array(keypoint.pt) - np.array(keypoint_list_l[index].pt)))
    radius_l.append(min(distance[keypoint_i]))
    # print(keypoint_i, " radius_l:", radius_l[keypoint_i])
    keypoint_i = keypoint_i + 1

# right keypoints
keypoint_list_r = []
for i, keypoint in enumerate(kp_r):
    keypoint_list_r.append(keypoint)

# sort by response
keypoint_list_r.sort(key=cmpfun, reverse=True)

# find minimum
distance = []
radius_r = []
keypoint_i = 0
for keypoint in keypoint_list_r:
    # print("Keypoint:", keypoint.response)
    distance.append([])
    if keypoint_i == 0:
        distance[0].append(1)
    for index in range(keypoint_i):
        distance[keypoint_i].append(np.linalg.norm(np.array(keypoint.pt) - np.array(keypoint_list_r[index].pt)))
    radius_r.append(min(distance[keypoint_i]))
    # print(keypoint_i, " radius_r:", radius_r[keypoint_i])
    keypoint_i = keypoint_i + 1

# sort by suppression radius
keypoint_list_l = np.c_[keypoint_list_l, radius_l]
keypoint_list_l = sorted(keypoint_list_l, key=lambda x:x[1], reverse=True)

keypoint_list_r = np.c_[keypoint_list_r, radius_r]
keypoint_list_r = sorted(keypoint_list_r, key=lambda x:x[1], reverse=True)

topn = 50  # get top n = 50
keypoint_list_l = keypoint_list_l[0:topn]
keypoint_list_l = np.delete(keypoint_list_l, 1, axis=1).transpose()[0]

keypoint_list_r = keypoint_list_r[0:topn]
keypoint_list_r = np.delete(keypoint_list_r, 1, axis=1).transpose()[0]

#img3_l = cv2.drawKeypoints(gray_l, keypoint_list_l, None, color=(0,255,0), flags=0)
#img3_r = cv2.drawKeypoints(gray_r, keypoint_list_r, None, color=(0,255,0), flags=0)
#plt.imsave("../../output/task_7/l_suppressed_key_points.png", img3_l)
#plt.imsave("../../output/task_7/r_suppressed_key_points.png", img3_r)
#print(keypoint_list)
#print(kp_l)


# Match features
keypoint_list_l, des_l = orb.compute(gray_l, keypoint_list_l)
keypoint_list_r, des_r = orb.compute(gray_r, keypoint_list_r)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(des_l, des_r)
img4 = cv2.drawMatches(gray_l, keypoint_list_l, gray_r, keypoint_list_r, matches, img_l)
plt.imsave("../../output/task_7/matches.png", img4)

# step 3: Calculate the essential matrix between the image pair.
pts_l = []
pts_r = []
for pts in matches:
    pts_l.append(keypoint_list_l[pts.queryIdx].pt)
    pts_r.append(keypoint_list_r[pts.trainIdx].pt)
pts_l = np.array(pts_l)
pts_r = np.array(pts_r)

projMatrix_l = np.eye(3)

F, mask = cv2.findFundamentalMat(pts_l, pts_r, cv2.FM_LMEDS)
E, mask = cv2.findEssentialMat(pts_l, pts_r, projMatrix_l, cv2.RANSAC)
pts_l = pts_l[mask.ravel() == 1]
pts_r = pts_r[mask.ravel() == 1]

matches = []
kp_l = []
kp_r = []
for pt_idx in range(len(pts_l)):
    matches.append(cv2.DMatch(pt_idx, pt_idx, 0))
    kp_l.append(cv2.KeyPoint(pts_l[pt_idx][0], pts_l[pt_idx][1], 0, 0))
    kp_r.append(cv2.KeyPoint(pts_r[pt_idx][0], pts_r[pt_idx][1], 0, 0))
img5 = cv2.drawMatches(gray_l, kp_l, gray_r, kp_r, matches, img_l)
plt.imsave("../../output/task_7/essential_matches.png", img5)

'''
lines = cv2.computeCorrespondEpilines(pts_l.reshape(-1, 1, 2), 2, F)
lines = lines.reshape(-1, 3)
img5,img6 = drawlines(gray_l, gray_r, lines, pts_l, pts_r)
plt.imshow(img5);plt.show()
'''

# step 4: Determine the relative camera pose from the essential matrix.
ret, R, t, mask1 = cv2.recoverPose(E, pts_l, pts_r)

# step 5: Check the reconstruction results

