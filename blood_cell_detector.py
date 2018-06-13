import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Read all XML files
path = 'BCCD_Dataset'
xml_files = [(os.path.join(root, name))
	for root, dirs, files in os.walk(path)
	for name in files if name.endswith((".xml"))]

# HOG parametrization
winSize = (64,64)
blockSize = (16,16)
blockStride = (4,4)
cellSize = (8,8)
nbins = 9
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
useSignedGradients = True
 
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,
	cellSize,nbins,derivAperture,winSigma,histogramNormType
	,L2HysThreshold,gammaCorrection,nlevels, useSignedGradients)

# Retrieve image patches from XML info and images
features = np.zeros((1,6084),np.float32)
labels = np.zeros(1,np.int64)
for t in xml_files:
	root = ET.parse(t).getroot()
	img_name = root[1].text
	img = cv2.imread(path+'/'+img_name)
	#hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	for elem in root.findall('object'):
		classes = elem[0].text
		x1 = int(elem[4][0].text)
		y1 = int(elem[4][1].text)
		x2 = int(elem[4][2].text)
		y2 = int(elem[4][3].text)
		if x2 > x1 and y2 > y1:
			if classes == 'RBC':
				labels = np.vstack((labels, 0))
			elif classes == 'WBC':
				labels = np.vstack((labels, 1))
			elif classes == 'Platelets':
				labels = np.vstack((labels, 2))

			cropped_img = img[y1:y2, x1:x2]
			resized_img = cv2.resize(cropped_img, winSize)
			descriptor = np.transpose(hog.compute(resized_img))
			features = np.vstack((features, descriptor))
			print img_name, classes, x1, y1, x2, y2
			#cv2.imshow(classes, resized_img)
			#k = cv2.waitKey(0)
			#if k == ord('q'):
			#	break

features = np.delete(features, (0), axis=0)
labels = np.delete(labels, (0), axis=0).ravel()

#print features.shape, labels.shape

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

print X_train.shape, y_train.shape
print X_test.shape, y_test.shape

clf = svm.SVC()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print 'Accuracy: ', accuracy_score(y_test, y_pred)
'''
img_files = [(os.path.join(root, name))
	for root, dirs, files in os.walk(path)
	for name in files if name.endswith((".jpg"))]

images = [cv2.imread(i) for i in img_files]

for i in images:
	cv2.imshow('ORIGINAL', i)
	k = cv2.waitKey(0)
	if k == ord('q'):
		break
'''