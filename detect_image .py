# USAGE
# python detect_mask_image.py --image examples/example_01.png

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

import tensorflow as tf
import argparse
import _thread
import pickle
import src.align.detect_face
import collections
import src.facenet

# construct the argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--path', help='Path of the video you want to test on.', default=0)
args = parser.parse_args()
MINSIZE = 20
THRESHOLD = [0.6, 0.7, 0.7]
FACTOR = 0.709
IMAGE_SIZE = 182
INPUT_IMAGE_SIZE = 160
CLASSIFIER_PATH = 'Models/facemodel.pkl'
VIDEO_PATH = args.path
FACENET_MODEL_PATH = 'Models/20180402-114759.pb'
with open(CLASSIFIER_PATH, 'rb') as file:
	modelf, class_names = pickle.load(file)
print("Custom Classifier, Successfully loaded")
with tf.Graph().as_default():
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
	sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

	with sess.as_default():
		# Load the model
		print('Loading feature extraction model')
		src.facenet.load_model(FACENET_MODEL_PATH)

		# Get input and output tensors
		images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
		embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
		phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
		embedding_size = embeddings.get_shape()[1]

		pnet, rnet, onet = src.align.detect_face.create_mtcnn(sess, "src/align")

		people_detected = set()
		person_detected = collections.Counter()

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = r"face_detector/deploy.prototxt"
weightsPath = r"face_detector/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
model = load_model("/home/thopham/Documents/Year3/KyI/PBL4/pbl4_final/Models/mask_detector.model")

# load the input image from disk, clone it, and grab the image spatial
# dimensions

#THAY DUONG LINK ANH
image = cv2.imread("/home/thopham/Pictures/Images_test_AI_ThoPham/with_outmask/nghieng_phai.jpg")
orig = image.copy()
(h, w) = image.shape[:2]

# construct a blob from the image
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
	(104.0, 177.0, 123.0))

# pass the blob through the network and obtain the face detections
print("[INFO] computing face detections...")
net.setInput(blob)
detections = net.forward()

# loop over the detections
for i in range(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with
	# the detection
	confidence = detections[0, 0, i, 2]

	# filter out weak detections by ensuring the confidence is
	# greater than the minimum confidence

	#THAY ĐỔI CHỈ SỐ TỪ 0.2-0.5
	if confidence > 0.2:
		# compute the (x, y)-coordinates of the bounding box for
		# the object
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		# ensure the bounding boxes fall within the dimensions of
		# the frame
		(startX, startY) = (max(0, startX), max(0, startY))
		(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

		# extract the face ROI, convert it from BGR to RGB channel
		# ordering, resize it to 224x224, and preprocess it
		face = image[startY:endY, startX:endX]
		face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
		face = cv2.resize(face, (224, 224))
		face = img_to_array(face)
		face = preprocess_input(face)
		face = np.expand_dims(face, axis=0)

		# pass the face through the model to determine if the face
		# has a mask or not
		(mask, withoutMask) = model.predict(face)[0]

		# determine the class label and color we'll use to draw
		# the bounding box and text
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)


		if label == "Mask":
			# include the probability in the label
			label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
			print("check khau trang")
			# display the label and bounding box rectangle on the output
			# frame
			cv2.putText(image, label, (startX, startY - 10),
						cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
		else:

			print("check face")

			# if faces_found > 1:
			#     cv2.putText(frame, "Only one face", (0, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL,
			#                 1, (255, 255, 255), thickness=1, lineType=2)

			# det = bounding_boxes[:, 0:4]q
			bb = np.zeros((1, 4), dtype=np.int32)

			(startX, startY, endX, endY) = box
			bb[0][0] = startX
			bb[0][1] = startY
			bb[0][2] = endX
			bb[0][3] = endY
			print(bb[0][3] - bb[0][1])
			print(image.shape[0])
			print((bb[0][3] - bb[0][1]) / image.shape[0])
			if (bb[0][3] - bb[0][1]) / image.shape[0] > 0.25:
				cropped = image[bb[0][1]:bb[0][3], bb[0][0]:bb[0][2], :]
				scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
									interpolation=cv2.INTER_CUBIC)
				scaled = src.facenet.prewhiten(scaled)
				scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
				feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
				emb_array = sess.run(embeddings, feed_dict=feed_dict)

				predictions = modelf.predict_proba(emb_array)
				best_class_indices = np.argmax(predictions, axis=1)
				best_class_probabilities = predictions[
					np.arange(len(best_class_indices)), best_class_indices]
				best_name = class_names[best_class_indices[0]]
				print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))

				if best_class_probabilities > 0.5:
					cv2.rectangle(image, (bb[0][0], bb[0][1]), (bb[0][2], bb[0][3]), (0, 255, 0), 2)
					text_x = bb[0][0]
					text_y = bb[0][3] + 20

					name = class_names[best_class_indices[0]]
					cv2.putText(image, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
								1, (255, 255, 255), thickness=1, lineType=2)
					cv2.putText(image, str(round(best_class_probabilities[0], 3)),
								(text_x, text_y + 17),
								cv2.FONT_HERSHEY_COMPLEX_SMALL,
								1, (255, 255, 255), thickness=1, lineType=2)
					person_detected[best_name] += 1
				else:
					name = "Unknown"



# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)