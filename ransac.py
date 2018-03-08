import cv2
import numpy as np
from random import randint
import math

def dlt(srcp, dstp):
	src_avgx,src_avgy=0,0
	dst_avgx,dst_avgy=0,0
	for i in range(len(srcp)):
		src_avgx+=srcp[i][0]
		src_avgy+=srcp[i][1]
		dst_avgx+=dstp[i][0]
		dst_avgy+=dstp[i][1]
	src_avg=[src_avgx/len(srcp),src_avgy/len(srcp)]
	dst_avg=[dst_avgx/len(dstp),dst_avgy/len(dstp)]
	src_dist,dest_dist = 0,0
	for i in range(0,len(srcp)):
		src_dist += math.sqrt(((srcp[i][0]-src_avg[0])**2)+((srcp[i][1]-src_avg[1])**2))
		dest_dist += math.sqrt(((dstp[i][0]-dst_avg[0])**2)+((dstp[i][1]-dst_avg[1])**2))
	scale_src = (math.sqrt(2))/(src_dist/len(srcp))
	scale_dst = (math.sqrt(2))/(dest_dist/len(dstp))
	t_src = np.asarray([[scale_src, 0, -src_avg[0]*scale_src],[0,scale_src,-src_avg[1]*scale_src], [0,0,1]])
	t_dst = np.asarray([[scale_dst, 0, -dst_avg[0]*scale_dst],[0,scale_dst,-dst_avg[1]*scale_dst], [0,0,1]])
	for i in range(len(srcp)):
		q,w    = srcp[i]
		srcp_n = np.dot(t_src,[q,w,1])
		srcp[i]= [srcp_n[0]/srcp_n[2],srcp_n[1]/srcp_n[2]]

		x,y    = dstp[i]
		dstp_n = np.dot(t_dst,[x,y,1])
		dstp[i]= [dstp_n[0]/dstp_n[2],dstp_n[1]/dstp_n[2]]
	a = np.zeros(shape=(len(srcp)*2,9))
	counter = 0
	i = 0
	while i<len(srcp)*2:
		a[i] = [0,0,0,-srcp[counter][0],-srcp[counter][1],-1, dstp[counter][1]*srcp[counter][0],dstp[counter][1]*srcp[counter][1],dstp[counter][1]]
		a[i+1] = [srcp[counter][0],srcp[counter][1],1,0,0,0,-dstp[counter][0]*srcp[counter][0],-dstp[counter][0]*srcp[counter][1],-dstp[counter][0]]
		counter+=1
		i+=2
	u,s,vt = cv2.SVDecomp(a)
	h = vt[-1].reshape(3,3)
	h_half = np.dot(np.linalg.inv(t_dst),h)
	h_final=np.dot(h_half,t_src)
	return h_final
def ransac(srcp,dstp):
	n = 10000
	i = 0
	inliers = 0
	inlier_ratio=0
	final_homograpy = np.zeros((3,3))
	final_inliers_src = []
	final_inliers_dst = []
	while i<n:
		inlier_points_src = []
		inlier_points_dst = []
		rand_src = []
		rand_dst = []
		for j in range(4):
			x = randint(0, len(srcp)-1)
			rand_src.append(srcp[x])
			rand_dst.append(dstp[x])
		homography = dlt(rand_src, rand_dst)
		for k in range(len(srcp)):
			srcpx, srcpy = srcp[k][0],srcp[k][1]
			x_prime = np.dot(homography,[srcpx,srcpy,1])
			x_x, x_y= x_prime[0]/x_prime[2], x_prime[1]/x_prime[2]
			for l in range(len(dstp)):
				if math.sqrt((x_x-dstp[l][0])**2+(x_y-dstp[l][1])**2)<=3:
					inlier_points_src.append(srcp[k])
					inlier_points_dst.append(dstp[k])
					inliers+=1
					break
		if(inlier_ratio<((100*inliers)/len(srcp))):
			inlier_ratio = (100*inliers)/len(srcp)
			final_inliers_src = inlier_points_src
			final_inliers_dst = inlier_points_dst
			final_homograpy = np.asarray(homography)
		w_s = (inliers/len(srcp))**4
		if(np.floor(abs(np.log(0.01)/np.log(abs(1-w_s))))>n):
			i+=1
			continue
		n=np.floor(abs(np.log(0.01)/np.log(abs(1-w_s))))
		inliers=0
		i+=1
	return final_inliers_src,final_inliers_dst

def ransac_while(srcp,dstp,homography):
	n = 10000
	i = 0
	inliers = 0
	inlier_ratio=0
	final_homograpy = np.zeros((3,3))
	final_inliers_src = []
	final_inliers_dst = []
	while i<n:
		inlier_points_src = []
		inlier_points_dst = []
		rand_src = []
		rand_dst = []
		for j in range(4):
			x = randint(0, len(srcp)-1)
			rand_src.append(srcp[x])
			rand_dst.append(dstp[x])
		for k in range(len(srcp)):
			srcpx, srcpy = srcp[k][0],srcp[k][1]
			x_prime = np.dot(homography,[srcpx,srcpy,1])
			x_x, x_y= x_prime[0]/x_prime[2], x_prime[1]/x_prime[2]
			for l in range(len(dstp)):
				if math.sqrt((x_x-dstp[l][0])**2+(x_y-dstp[l][1])**2)<=3:
					inlier_points_src.append(srcp[k])
					inlier_points_dst.append(dstp[k])
					inliers+=1
					break
		if(inlier_ratio<((100*inliers)/len(srcp))):
			inlier_ratio = (100*inliers)/len(srcp)
			final_inliers_src = inlier_points_src
			final_inliers_dst = inlier_points_dst
			final_homograpy = np.asarray(homography)
		w_s = (inliers/len(srcp))**4
		if(np.floor(abs(np.log(0.01)/np.log(abs(1-w_s))))>n):
			i+=1
			continue
		n=np.floor(abs(np.log(0.01)/np.log(abs(1-w_s))))
		inliers=0
		i+=1
	return final_inliers_src,final_inliers_dst
if __name__ == '__main__':
	img_src =cv2.imread("img1.jpg")
	img_dst =cv2.imread("img2.jpg")
	orb = cv2.ORB_create(nfeatures=5000)
	kp_src,des_src= orb.detectAndCompute(img_src,None)
	kp_dst, des_dst=orb.detectAndCompute(img_dst,None)
	print("Keypoints found.")
	bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)
	matches = bf.match(des_src,des_dst)
	dmatches = sorted(matches, key = lambda x:x.distance)
	print("Keypoints matched.")
	src_pts  = np.float32([kp_src[m.queryIdx].pt for m in dmatches])
	dst_pts  = np.float32([kp_dst[m.trainIdx].pt for m in dmatches])
	inliers_src, inliers_dst = ransac(src_pts, dst_pts)
	print("RANSAC finished. Found inlier number is:", len(inliers_src))
	inlier_counter=0
	while True:
		inlier_threshold=len(inliers_src)
		dlt_final_homography = dlt(inliers_src, inliers_dst)
		print("Found homography: \n", dlt_final_homography)
		inliers_src, inliers_dst = ransac_while(inliers_src,inliers_dst,dlt_final_homography)
		print("Reduced inlier count:", len(inliers_src))
		if(len(inliers_src)==0):
			print("No inlier found exited.")
			break
		print("Inlier converge calc started found inliers:" ,len(inliers_src))
		if(inlier_threshold-len(inliers_src)<7):
			inlier_counter+=1
		else:
			inlier_counter=0
		if inlier_counter>4:
			dlt_final_homography = dlt(inliers_src, inliers_dst)
			break
	rec_point1 = np.dot(dlt_final_homography,[175,576,1])
	rec_point1 = [abs(rec_point1[1]),abs(rec_point1[0])]
	print("Point 1:", rec_point1)
	rec_point2 = np.dot(dlt_final_homography,[155,470,1])
	rec_point2 = [abs(rec_point2[1]),abs(rec_point2[0])]
	print("Point 2:", rec_point2)
	img3 = cv2.rectangle(img_dst,(int(rec_point1[0]),int(rec_point1[1])),(int(rec_point2[0]),int(rec_point2[1])),(0,0,255),2)

	while(1):
	    k =  cv2.waitKey(1) & 0xFF
	    cv2.imshow("ROI",img3)
	    if(k==27):
	        cv2.destroyAllWindows()
	        break
