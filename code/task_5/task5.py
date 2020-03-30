import cv2
import numpy as np

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Loading image
#img_l = cv2.imread('../../images/task_5/right_0.png')
#img_r = cv2.imread('../../images/task_5/right_0.png')
def planarHomography(imgPath, indicator):
    img_l = cv2.imread(imgPath)
    m = np.array([[8, 0, 0], [0, 8, 0], [0, 0, 1]]).T.reshape(3,3)
    n = np.array([[2,5,9], [1,7,3], [2,7,3], [3,5,3]])
    #print(m)
    objp = np.zeros((9 * 6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    objp = np.dot(objp,m)

    for x in range(54):
        objp[x][0] += 200
        objp[x][1] += 400

    objpoints = []  # 2d point in real world space
    imgpoints = []  # 2d points in image plane.

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
        #cv2.waitKey(0)
        #print(corners)


    # Undistort left image
    h, w = img_l.shape[:2]
    mapx, mapy = cv2.initUndistortRectifyMap(
        cameraMatrix_l.mat(), distMatrix_l.mat(), None, projMtx1.mat(), (w, h), 5)
    dst = cv2.remap(img_l, mapx, mapy, cv2.INTER_LINEAR)
    if indicator is "l0":
        cv2.imwrite("../../output/task_5/undistored_left_0.png", dst)
    else:
        cv2.imwrite("../../output/task_5/undistored_left_1.png", dst)


    # Finding homography
    h, status = cv2.findHomography(corners2, objp)
    print(h)
    img_warp = cv2.warpPerspective(dst, h, (dst.shape[1], dst.shape[0]))
    if indicator is "l0":
        cv2.imwrite("../../output/task_5/warped_left_0.png", img_warp)
    else:
        cv2.imwrite("../../output/task_5/warped_left_1.png", img_warp)


def main():
    '''
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-I1', "--image1", help="Path to the left image", default="../../images/task_5/left_0.png")
    parser.add_argument('-I2', "--image2", help="Path to the right image", default="../../images/task_5/right_0.png")
    args = parser.parse_args()

    img_l = args.image1
    img_r = args.image2
    '''
    img_l1 = "../../images/task_5/left_0.png"
    img_l2 = "../../images/task_5/left_1.png"
    planarHomography(img_l1, "l0")
    planarHomography(img_l2, "l1")

if __name__ == "__main__":
    main()
