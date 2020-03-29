import cv2 as cv
import numpy as np

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

img = cv.imread('../../images/task_5/left_0.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

objp = np.zeros((9 * 6, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

ret, corners = cv.findChessboardCorners(gray, (9, 6))
objpoints.append(objp)
corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
imgpoints.append(corners2)
img = cv.drawChessboardCorners(img, (9, 6), corners2, ret)
#cv.imshow("Corners", img)
# cv.waitKey(0)

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

mapx, mapy = cv.initUndistortRectifyMap(
    mtx, dist, None, mtx, (w, h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]
#cv.imwrite(r'../../output/task_5/output.png', dst)
cv.imshow("Undistorted", dst)
cv.waitKey(0)

'''
H, _ = cv.findHomography(corners, objp)
# print(H)
img_warp = cv.warpPerspective(img, H, (img.shape[1], img.shape[0]))
#cv.imshow("Warped", img_warp)
cv.imwrite(r'../../output/task_5/l_warped.png', img_warp)
'''
