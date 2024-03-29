import numpy as np
import cv2
import scipy.signal as sgn
from extract_boite_from_Video import extract_boite_from_video


def construct_video_patches(video, nx_boxes, ny_boxes): 
	# Build patch videos
	# TODO: make step proportional to resolution
	videos = []
	#plot=np.copy(video[0]) #
	ci = 15
	for i in range(ny_boxes):
	    cj=0
	    for j in range(nx_boxes):
	        cj+=10
	        videos+=[video[:,ci+15*i:ci+15*(i+1),:][:,:,cj+15*j:cj+15*(j+1)]]
	        #plot[ci+15*i:ci+15*(i+1),:][:,cj+15*j:cj+15*(j+1)]=[[[0,0,255] for h in range(15)]for k in range(15)] #
	    ci+=10
	return videos


def build_video_array(avi_filename):
	cap   = cv2.VideoCapture(avi_filename)
	video = []
	while(cap.isOpened()): # play the video by reading frame by frame
	        ret, frame = cap.read()
	        if ret==True:
	            video+=[frame]
	        else:
	            break
	cap.release()
	return np.asarray(video)


def H_matrix(n, Lambda):
	# detrended using smoothness priors approach
	I  = np.eye(n)                 # Return a 2-D array with ones on the diagonal and zeros elsewhere, n Number of rows
	D2 = np.zeros((n-2,n))
	for i in range(n-2):
	    D2[i,i]   = 1
	    D2[i,i+1] = -2
	    D2[i,i+2] = 1
	    
	return I - np.linalg.inv(I + Lambda**2*D2.T.dot(D2))


def preprocessing(input_video, output_filename, avi_output_filename, landmarks_file, nx_boxes=4, ny_boxes=5, Lambda=10, fN=10):

	n_boxes = nx_boxes * ny_boxes

	extract_boite_from_video(input_video, landmarks_file, avi_output_filename) 

	video = build_video_array(avi_output_filename)

	videos = construct_video_patches(video, nx_boxes, ny_boxes)

	# Extracts channels for each video
	videos_b = []
	videos_g = []
	videos_r = []
	for vid in videos:
	    videos_b += [vid[:,:,:,0]]
	    videos_g += [vid[:,:,:,1]]
	    videos_r += [vid[:,:,:,2]]


	# Calculate mean values for each box at each frame

	videos_b_mean = [[] for i in range(n_boxes)]
	videos_g_mean = [[] for i in range(n_boxes)]
	videos_r_mean = [[] for i in range(n_boxes)]

	for j in range(n_boxes):
	    for i in range(videos[0].shape[0]):
	        videos_b_mean[j] += [np.mean(videos_b[j][i],axis=(0,1))]
	        videos_g_mean[j] += [np.mean(videos_g[j][i],axis=(0,1))]
	        videos_r_mean[j] += [np.mean(videos_r[j][i],axis=(0,1))]


	# Subtract mean value per box on all frames

	videos_b_mean = np.array(videos_b_mean)
	videos_g_mean = np.array(videos_g_mean)
	videos_r_mean = np.array(videos_r_mean)

	videos_b_zero_mean = (videos_b_mean.T - np.mean(videos_b_mean, axis=1)).T   # zero mean
	videos_g_zero_mean = (videos_g_mean.T - np.mean(videos_g_mean, axis=1)).T
	videos_r_zero_mean = (videos_r_mean.T - np.mean(videos_r_mean, axis=1)).T


    # Define H

	n = videos_b_zero_mean[0].shape[0] # number of frames
	H = H_matrix(n, Lambda)


	# Filter

	fN=10 #fe = 20 Hz et donc f_Nyquist = fe/2 = 10 Hz
	b, a = sgn.butter(2, (0.7/fN,3.5/fN),'bandpass')  #frequences normalisees


	# Apply H

	videos_b_f = []
	videos_g_f = []
	videos_r_f = []

	for i in range(n_boxes):
	    videos_b_f += [H.dot(videos_b_zero_mean[i])]
	    videos_g_f += [H.dot(videos_g_zero_mean[i])]
	    videos_r_f += [H.dot(videos_r_zero_mean[i])]
    
    
	# Apply filter

	videos_B = []
	videos_G = []
	videos_R = []

	for i in range(n_boxes):
	    
	    videos_B += [sgn.filtfilt(b, a, videos_b_f[i])]
	    videos_G += [sgn.filtfilt(b, a, videos_g_f[i])]
	    videos_R += [sgn.filtfilt(b, a, videos_r_f[i])]

	# Save to numpy file
	data = np.asarray(videos_B + videos_G + videos_R)
	np.save(output_filename, data.T)
        
	#return (np.asarray(videos_B), np.asarray(videos_G), np.asarray(videos_R))




import os
import glob

# local
#data_path = './Dataset/'

# Jean Zay
#data_path = '/gpfswork/rech/qbf/commun/Dataset/'

# LML
data_path = '/media/data/UBFG/DATASET_2/'

folders = sorted(glob.glob(data_path+'subject9'))
print('folders = ', folders)

for subject_folder in folders:
	print('Preprocessing ', subject_folder)
	preprocessing(input_video=subject_folder+'/vid.avi', landmarks_file=data_path+'shape_predictor_5_face_landmarks.dat', output_filename=subject_folder+'/signals.npy', avi_output_filename=subject_folder+'/out.avi')
   
