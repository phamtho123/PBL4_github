# import system module
import sys

# import some PyQt5 modules
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer

# import Opencv module

from GUI.ui_main_window import *
from tensorflow.keras.models import load_model
import numpy as np
import cv2

import tensorflow as tf
import argparse
import pickle
import src.align.detect_face
import src.detect_and_mask
import collections
import src.facenet

import sqlite3
import csv
import time
from openpyxl import Workbook
from xlsxwriter.workbook import Workbook

con = sqlite3.connect('QuanLi.db')

"""cursorObj = con.cursor()
cursorObj.execute("CREATE TABLE employees(ID INTEGER PRIMARY KEY AUTOINCREMENT, Name TEXT, Time TEXT)")
con.commit()"""

def Add(con,name,time):
    cursorObj = con.cursor()
    cursorObj.execute("insert into employees (Name, Time) VALUES (?,?)", (name, time))
    con.commit()


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
id = 0
count = 1
with open(CLASSIFIER_PATH, 'rb') as file:
    model, class_names = pickle.load(file)
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
prototxtPath = r"face_detector/deploy.prototxt"
weightsPath = r"face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("Models/mask_detector.model")

# initialize the video stream
print("[INFO] starting video stream...")

class MainWindow(QWidget):
    # class constructor
    def __init__(self):
        # call QWidget constructor
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        # create a timer
        self.timer = QTimer()
        # set timer timeout callback function
        self.timer.timeout.connect(self.viewCam)
        # set btnStart callback clicked  function
        # khi mà nhấn start là đưa đến hàm controltimer
        self.ui.btnStart.clicked.connect(self.controlTimer)
        self.ui.btnXuatfile.clicked.connect(self.Xuatfile)

        self.ui.btnExit.clicked.connect(self.thoat)

    # view camera
    def viewCam(self):
    # read image in BGR format
        ret, frame = self.cap.read()
        frame = cv2.flip(frame, 1)

        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = src.detect_and_mask.detect_and_predict_mask(frame, faceNet, maskNet)

        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            if label == "Mask":
                # include the probability in the label
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                # display the label and bounding box rectangle on the output
                # frame

                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            else:
                try:
                    bb = np.zeros((1, 4), dtype=np.int32)
                    (startX, startY, endX, endY) = box
                    bb[0][0] = startX
                    bb[0][1] = startY
                    bb[0][2] = endX
                    bb[0][3] = endY
                    print(bb[0][3] - bb[0][1])
                    print(frame.shape[0])
                    print((bb[0][3] - bb[0][1]) / frame.shape[0])
                    if (bb[0][3] - bb[0][1]) / frame.shape[0] > 0.25:
                        cropped = frame[bb[0][1]:bb[0][3], bb[0][0]:bb[0][2], :]
                        scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                            interpolation=cv2.INTER_CUBIC)
                        scaled = src.facenet.prewhiten(scaled)
                        scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                        feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                        emb_array = sess.run(embeddings, feed_dict=feed_dict)

                        predictions = model.predict_proba(emb_array)
                        best_class_indices = np.argmax(predictions, axis=1)
                        best_class_probabilities = predictions[
                            np.arange(len(best_class_indices)), best_class_indices]
                        best_name = class_names[best_class_indices[0]]
                                #print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))
                        named_tuple = time.localtime()  # lấy struct_time
                        # time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
                        time_string = time.strftime("%m/%d, %H:%M", named_tuple)
                        Add(con,best_name,time_string)
                        global  count
                        if count % 60 == 0 :
                            global id
                            person = {
                                "Name": best_name,
                                "Time": time_string
                            }
                            self.ui.tableWidget.setRowCount(1000000)
                            self.ui.tableWidget.setItem(id, 0, QtWidgets.QTableWidgetItem(person["Name"]))
                            self.ui.tableWidget.setItem(id, 1, QtWidgets.QTableWidgetItem(person["Time"]))
                            id += 1
                            print(str(id) + " " + best_name + " , " + time_string)


                        count+=1;

                        if best_class_probabilities > 0.5:
                            cv2.rectangle(frame, (bb[0][0], bb[0][1]), (bb[0][2], bb[0][3]), (0, 0, 255), 2)
                            text_x = bb[0][0]
                            text_y = bb[0][3] + 20

                            name = class_names[best_class_indices[0]]
                            cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1, (0, 0, 255), thickness=1, lineType=2)
                            cv2.putText(frame, str(round(best_class_probabilities[0], 3)),
                                        (text_x, text_y + 17),
                                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1, (0, 0, 255), thickness=1, lineType=2)
                            person_detected[best_name] += 1
                        else:
                            name = "Unknown"
                except:
                    pass

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # get image infos
        height, width, channel = frame.shape
        step = channel * width
        # create QImage from image
        qImg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
        # show image in img_label
        self.ui.image_label.setPixmap(QPixmap.fromImage(qImg))


    def thoat(self):
        reply = QMessageBox.question(self, 'Window Close', 'Bạn có muốn thoát?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            exit()

    # start/stop timer
    def controlTimer(self):
        # if timer is stopped
        if not self.timer.isActive():
            # create video capture
            self.cap = cv2.VideoCapture(0)
            # start timer
            self.timer.start(20)
            # update btnStart text
            self.ui.btnStart.setText("Started")
        # if timer is starte
    def Xuatfile(self):
        cursorObj = con.cursor()
        data = cursorObj.execute('SELECT * FROM employees')
        with open("ListViPham.csv", "w") as csv_file:
            csv_writer = csv.writer(csv_file, delimiter="\t")
            csv_writer.writerow([i[0] for i in cursorObj.description])
            csv_writer.writerows(cursorObj)

        workbook = Workbook('ListViPham.xlsx')
        worksheet = workbook.add_worksheet()
        c = con.cursor()
        c.execute('SELECT * FROM employees')
        mysel = c.execute('SELECT * FROM employees')
        for i, row in enumerate(mysel):
            for j, value in enumerate(row):
                worksheet.write(i, j, value)
        workbook.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # create and show mainWindow
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())