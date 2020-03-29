import cv2
import numpy as np

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Loading image
img_l = cv2.imread('../../images/task_5/left_0.png')
img_r = cv2.imread('../../images/task_5/right_0.png')

objp = np.zeros((9 * 6, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
print(objp)
objpoints = []  # 2d point in real world space
imgpoints = []  # 2d points in image plane.

# print(objp)
# gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
# Loading parameters
fs_l = cv2.FileStorage(
    "../../parameters/left_camera_intrinsics.xml", cv2.FILE_STORAGE_READ)
cameraMatrix_l = fs_l.getNode("camera_intrinsic")
distMatrix_l = fs_l.getNode("distort_coefficients")

fs_projection = cv2.FileStorage(
    "../../parameters/stereo_rectification.xml", cv2.FILE_STORAGE_READ)
projMtx1 = fs_projection.getNode("rectified_projection_matrix_1")
projMtx2 = fs_projection.getNode("rectified_projection_matrix_2")


gray = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
# Find the chess board corners
ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
if ret:
    objpoints.append(objp)
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    imgpoints.append(corners2)

    # Draw and display the corners
    img = cv2.drawChessboardCorners(img_l, (9, 6), corners2, ret)
    #cv2.imwrite(r'../../output/task_5/l_calibrated.png', img)
    #cv2.imshow("Corners", img)
    # cv2.waitKey(0)


# Undistort left image
h, w = img_l.shape[:2]
mapx, mapy = cv2.initUndistortRectifyMap(
    cameraMatrix_l.mat(), distMatrix_l.mat(), None, projMtx1.mat(), (w, h), 5)
dst = cv2.remap(img_l, mapx, mapy, cv2.INTER_LINEAR)
# print(img)
#cv2.imshow("Undistort", dst)
# cv2.waitKey(0)
'''
x, y, w, h = roi_1
dst = dst[y:y + h, x:x + w]
cv2.imwrite(r'../../output/task_3/l_distort.png', dst)
'''

# Finding homography
'''
h, status = cv2.findHomography(dst, objp)
img_warp = cv2.warpPerspective(dst, h, (dst.shape[1], dst.shape[0]))
'''
