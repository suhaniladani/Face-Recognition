from facerec.feature import Fisherfaces
from facerec.classifier import NearestNeighbor
from facerec.model import PredictableModel
from facerec.serialization import save_model, load_model
import numpy as np
from PIL import Image
from PIL import ImageChops
import sys, os
import datetime
import time
import cv2
import multiprocessing

vc=cv2.VideoCapture('IMG_3348.MOV')
#vc=cv2.VideoCapture('IMG_35.mp4')
#vc=cv2.VideoCapture(0)

#queue = multiprocessing.Queue(maxsize=0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
def read_images(path, sz=(256,256)):
    """Reads the images in a given folder, resizes images on the fly if size is given.
    Args:
        path: Path to a folder with subfolders representing the subjects (persons).
        sz: A tuple with the size Resizes 
    Returns:
        A list [X,y]
            X: The images, which is a Python list of numpy arrays.
            y: The corresponding labels (the unique number of the subject, person) in a Python list.
    """
    c = 0
    X,y = [], []
    folder_names = []
    
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            folder_names.append(subdirname)
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    im = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)
                    # resize to given size (if given)
                    #if (sz is not None):
                    #    im = cv2.resize(im, sz)
                    #X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                except IOError, (errno, strerror):
                    print "I/O error({0}): {1}".format(errno, strerror)
                except:
                    print "Unexpected error:", sys.exc_info()[0]
                    raise
            c = c+1
    return [y,folder_names]	

pathdir='FaceDB_comp/FaceDB2/'
y,subject_names=read_images(pathdir)
list_of_labels = list(xrange(max(y)+1))
subject_dictionary = dict(zip(list_of_labels, subject_names))
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,0xff, 1 )
keyPressed=-1
counter=-1
temp=None
model=load_model('Training data1')
#prev_image = np.zeros((256,256), np.uint8)
start = time.time()
#queue=Queue(maxsize=0)
while (keyPressed < 0):
	rval, frame = vc.read()
	img = frame
	counter=counter+1
	if(counter%15==0):
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(
				gray,
				scaleFactor=1.3,
				minNeighbors=5,
				minSize=(30,30),
				flags=cv2.CASCADE_SCALE_IMAGE
				)
		temp=faces
			#	str_temp=str_temp,str(subject_dictionary[predicted_label])
			#queue.put(str(subject_dictionary[predicted_label]))
			#else:
			#str_t="Unrecognized Face"
			#queue.put(str_t)
		#temp=faces
		#cnt=0
		#cv2.putText(img,'Detected Face', (500,500), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,250),3,1)
		
	for (x,y,w,h) in temp:
		#cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		#roi_gray = gray[y:y+h, x:x+w];
		#roi_color = img[y:y+h, x:x+w]
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		mask = cv2.inRange(hsv, np.array((0,133,77),np.uint8), np.array((255,173,127),np.uint8))
		roi_hist = cv2.calcHist([hsv],[0],mask,[180],[0,180])
		cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
		dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
		track_window=x,y,w,h
		# apply meanshift to get the new location
		ret, track_window = cv2.CamShift(dst, track_window,term_crit)
		# Draw it on image
		sampleImage = gray[y:y+h, x:x+w]
		sampleImage = cv2.resize(sampleImage, (256,256))
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		[ predicted_label, generic_classifier_output] = model.predict(sampleImage)
		print [ predicted_label, generic_classifier_output]
		if int(generic_classifier_output['distances']) <=500:
			cv2.putText(img,'Hello,'+str(subject_dictionary[predicted_label]), (x,y), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,250),3,1)
		else:
			cv2.putText(img,'Unknown face', (x,y), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,250),3,1)
			
	cv2.imshow('result',img)
        keyPressed = cv2.waitKey(1) # wait 1 milisecond in each iteration of while loop
end=time.time()
fps=counter/(end-start)
print(fps)
cv2.destroyAllWindows()
vc.release()
