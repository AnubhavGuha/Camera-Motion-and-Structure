import numpy as np
import cv2
from cv2 import aruco
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Function for plot camera posture in 3D
def plotCamPos(mat_R,mat_t,h_rot,veri_rot,fn):
    # Set marker position
    markerP = [[0, 0, 1], [1, 1, 0], [1, -1, 0], [-1, -1, 0], [-1, 1, 0]]
    # Set figure, draw marker
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    square_marker = markerP
    square_marker.append(markerP[1])
    square_marker = np.transpose(square_marker[1:6])
    Axes3D.plot(ax, square_marker[0].flatten(), square_marker[1].flatten(), square_marker[2].flatten(), 'r')
    # For each figure, plot the camera posture
    for i in range(0,len(mat_R)):
        nowR = mat_R[i]
        nowt = mat_t[i]
        RTt = np.dot(np.transpose(nowR), nowt).transpose()
        nowCamP = []
        for point in markerP:
            pointCam = np.dot(np.transpose(nowR), point) - RTt
            nowCamP.append(pointCam)
        
        square_cam = nowCamP
        square_cam.append(nowCamP[1])
        square_cam = np.transpose(square_cam[1:6])
        line1_cam = np.transpose([nowCamP[1],nowCamP[0],nowCamP[2]])
        line2_cam = np.transpose([nowCamP[3],nowCamP[0],nowCamP[4]])
        # Plot
        Axes3D.plot(ax, square_cam[0].flatten(), square_cam[1].flatten(), square_cam[2].flatten(),'k')
        Axes3D.plot(ax, line1_cam[0].flatten(), line1_cam[1].flatten(), line1_cam[2].flatten(),'k')
        Axes3D.plot(ax, line2_cam[0].flatten(), line2_cam[1].flatten(), line2_cam[2].flatten(),'k')
        textP = nowCamP[0].flatten() 
        Axes3D.text(ax, x=textP[0,0], y=textP[0,1], z=textP[0,2], s=str(i), zdir=None)
    # Change view angle
    ax.view_init(veri_rot,h_rot)
    plt.savefig(fn)

# Main program starts here
outPath = '../../output/task_6/'

# Load figure
imgL = []
imgR = []
for i in range(0,11):
    tempImgL = cv2.imread('../../images/task_6/left_'+str(i)+'.png')
    tempImgR = cv2.imread('../../images/task_6/right_'+str(i)+'.png')
    imgL.append(tempImgL)
    imgR.append(tempImgR)

# Get something for aruco
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters =  aruco.DetectorParameters_create()
gray_th = 150 # For convert grayscale image to BW image, mainly sovle the problem of right-image-2
# For each figure of left and right camera, first left camera
# For left camera solvePnP usage
objPoints = np.array([[[0.0,0.0,0.0],[1.0,0.0,0.0],[1.0,1.0,0.0],[0.0,1.0,0.0]]])
camMatrixL = np.array([[423.27381306,0.0,341.34626532],[0.0,421.27401756,269.28542111],[0.0,0.0,1.0]])
distCoeL = np.array([-0.4339415742303,0.2670771755754,-0.0003114434702029,0.000563893810148,-0.1097045226614])
# Create list for R and t storage, left camera
R_L = []
t_L = []
for i in range(0,11):
    nowGrayImg = cv2.cvtColor(imgL[i], cv2.COLOR_BGR2GRAY)
    (th, nowBWImg) = cv2.threshold(nowGrayImg, gray_th, 255, cv2.THRESH_BINARY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(nowBWImg, aruco_dict, parameters=parameters)
    _,rvecs,tvecs = cv2.solvePnP(objectPoints=objPoints,imagePoints=corners[0],cameraMatrix=camMatrixL,distCoeffs=distCoeL)
    temp_R = cv2.Rodrigues(rvecs)[0]
    temp_t = -np.matrix(temp_R).T * np.matrix(tvecs)
    R_L.append(temp_R)
    t_L.append(temp_t*5)
    # Choose some images to save the marker detection results
    if((i==3) | (i==5)):
    	plt.figure()
    	frame_markers = aruco.drawDetectedMarkers(imgL[i].copy(), corners, ids)
    	plt.imshow(frame_markers)
    	plt.savefig(outPath+'markerDetectRes_leftCam_'+str(i)+'.png')
# Then for right camera
camMatrixR = np.array([[420.91160482,0.0,352.16135589],[0.0,418.72245958,264.50726699],[0.0,0.0,1.0]])
distCoeR = np.array([-0.41458176811769,0.199612732468976,-0.000148320911416565,-0.00136867604379664,-0.051135846250151])
# Create list for R and t storage, right camera
R_R = []
t_R = []
for i in range(0,11):
    nowGrayImg = cv2.cvtColor(imgR[i], cv2.COLOR_BGR2GRAY)
    (th, nowBWImg) = cv2.threshold(nowGrayImg, gray_th, 255, cv2.THRESH_BINARY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(nowBWImg, aruco_dict, parameters=parameters)
    _,rvecs,tvecs = cv2.solvePnP(objectPoints=objPoints,imagePoints=corners[0],cameraMatrix=camMatrixR,distCoeffs=distCoeR)
    temp_R = cv2.Rodrigues(rvecs)[0]
    temp_t = -np.matrix(temp_R).T * np.matrix(tvecs)
    R_R.append(temp_R)
    t_R.append(temp_t*5)
    if((i==4)|(i==7)|(i==10)):
    	plt.figure()
    	frame_markers = aruco.drawDetectedMarkers(imgR[i].copy(), corners, ids)
    	plt.imshow(frame_markers)
    	plt.savefig(outPath+'markerDetectRes_rightCam_'+str(i)+'.png')

# Plot camera posture
plotCamPos(R_L,t_L,120,30,outPath+'leftCamPos_1.png')
plotCamPos(R_L,t_L,240,60,outPath+'leftCamPos_2.png')
plotCamPos(R_L,t_L,60,30,outPath+'leftCamPos_3.png')
plotCamPos(R_L,t_L,180,30,outPath+'leftCamPos_4.png')
plotCamPos(R_L,t_L,300,30,outPath+'leftCamPos_5.png')
plotCamPos(R_L,t_L,60,60,outPath+'leftCamPos_6.png')

plotCamPos(R_R,t_R,120,30,outPath+'rightCamPos_1.png')
plotCamPos(R_R,t_R,240,60,outPath+'rightCamPos_2.png')
plotCamPos(R_R,t_R,60,30,outPath+'rightCamPos_3.png')
plotCamPos(R_R,t_R,180,30,outPath+'rightCamPos_4.png')
plotCamPos(R_R,t_R,300,30,outPath+'rightCamPos_5.png')
plotCamPos(R_R,t_R,60,60,outPath+'rightCamPos_6.png')