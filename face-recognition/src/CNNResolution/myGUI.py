import wx
import cv2

import tensorflow as tf
from BuildNet import NetworkBuilder
import numpy as np


class VideoPanel(wx.Panel):
    def __init__(self, parent, capture,recognizer,faceCascade):
        wx.Panel.__init__(self, parent)

        self.capture = capture
        self.recognizer = recognizer
        self.faceCascade = faceCascade
        #cnn
        self.sess,self.input_img,self.prediction = self.CNN()
        self.id=(1511278,1511296,1511298,1511308,1511333,
                 1511334,1511340,1511351,1511360,1511363)


        ret, img = self.capture.read()
        height, width = img.shape[:2]
        parent.SetSize((width, height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # self.bmp = wx.BitmapFromBuffer(width, height, frame)
        self.bmp = wx.Bitmap.FromBuffer(width, height, img)
        self.timer = wx.Timer(self)
        self.timer.Start(100)

        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_TIMER, self.NextFrame)

    def OnPaint(self, event):
        dc = wx.BufferedPaintDC(self)
        dc.DrawBitmap(self.bmp, 0, 0)

    def NextFrame(self, event):
        ret, img = self.capture.read()
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        if ret:

            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            faces = self.faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                global flag
                if True:
                     id, conf = self.recognizer.predict(gray[y:y + h, x:x + w])
                else :
                    img128 = []
                    img128.append(cv2.resize(img[y:y + h, x:x + w], (128,128)))
                    pre= self.sess.run([self.prediction], feed_dict={self.input_img: img128})
                    id = self.id[np.argmax(pre)]

                cv2.rectangle(img, (x - 10, y - 10), (x + w + 10, y + h + 10), (225, 0, 0), 2)
                cv2.rectangle(img, (x - 13, y - 10), (x + w + 10, y - 40), (0, 255, 0), 2)
                cv2.putText(img, str(id), (x, y - 20), font, 1, (255, 255, 255), 2)

            self.bmp.CopyFromBuffer(img)
            self.Refresh()

    def CNN(self):
        with tf.name_scope("Input") as scope:
            input_img = tf.placeholder(dtype='float32', shape=[None, 128, 128, 3], name="input")

        netBuild = NetworkBuilder()

        with tf.name_scope("Modelv2") as scope:
            model = input_img
            model = netBuild.attach_conv_layer(model, 32, summary=True)
            model = netBuild.attach_relu_layer(model)
            model = netBuild.attach_conv_layer(model, 32, summary=True)
            model = netBuild.attach_relu_layer(model)
            model = netBuild.attach_pooling_layer(model)

            model = netBuild.attach_conv_layer(model, 64, summary=True)
            model = netBuild.attach_relu_layer(model)
            model = netBuild.attach_conv_layer(model, 64, summary=True)
            model = netBuild.attach_relu_layer(model)
            model = netBuild.attach_pooling_layer(model)

            model = netBuild.attach_conv_layer(model, 128, summary=True)
            model = netBuild.attach_relu_layer(model)
            model = netBuild.attach_conv_layer(model, 128, summary=True)
            model = netBuild.attach_relu_layer(model)
            model = netBuild.attach_pooling_layer(model)

            model = netBuild.flatten(model)
            model = netBuild.attach_dense_layer(model, 200, summary=True)
            model = netBuild.attach_sigmod_layer(model)
            model = netBuild.attach_dense_layer(model, 32, summary=True)
            model = netBuild.attach_sigmod_layer(model)
            model = netBuild.attach_dense_layer(model, 10)

            prediction = netBuild.attach_soft_max_layer(model)


        saver = tf.train.Saver()
        sess = tf.Session()
        saver.restore(sess,tf.train.latest_checkpoint("Classifier/saved-model-v2/"))
        # with tf.Session() as sess:
        #     saver.restore(sess, tf.train.latest_checkpoint("Classifier/saved-model-v2/"))
        #     pre = sess.run([prediction], feed_dict={input_img: img128})

        return sess,input_img,prediction

class MyFrame(wx.Frame):
    def __init__(self, parent, title, capture,recognizer,faceCascade):
        wx.Frame.__init__(self, parent, title=title)
        self.videoPanel = VideoPanel(self, capture,recognizer,faceCascade)
        self.CenterOnScreen()
        # prepare the menu bar

        # 1st menu form left
        menu1 = wx.Menu()
        menu1.Append(101, "&Open From File")
        # menu1.Append(102,"&Open From Capture")
        menu1.Append(102, "&Exit")

        # 2st menu
        menu2 = wx.Menu()
        menu2.Append(201, "&LBPH")
        menu2.Append(202, "&CNN")
        # 3st menu
        menu3 = wx.Menu()
        menu3.Append(301, "&help")
        menu3.Append(302, "&reference")

        menuBar = wx.MenuBar()
        menuBar.Append(menu1, "&File")
        menuBar.Append(menu2, "&Run")
        menuBar.Append(menu3, "&About")
        self.SetMenuBar(menuBar)

        #menu events
        self.Bind(wx.EVT_MENU,self.lbph,id= 201)
        self.Bind(wx.EVT_MENU,self.cnn,id= 202)
    def cnn(self,event):
        global flag
        flag = False
    def lbph(self,event):
        global flag
        flag = True

#HLPB
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('Classifier/LBPH-trainer.xml')

faceCascade = cv2.CascadeClassifier("Classifier/haaracascade_frontalface_alt2.xml")

#CNN
# with tf.name_scope("Input") as scope:
#     input_img = tf.placeholder(dtype='float32', shape=[None, 128, 128, 3], name="input")
#
# with tf.name_scope("Target") as scope:
#     target_labels = tf.placeholder(dtype='float32', shape=[None, 10], name="Targets")
#
# netBuild = NetworkBuilder()
#
# with tf.name_scope("Modelv2") as scope:
#     model = input_img
#     model = netBuild.attach_conv_layer(model, 32, summary=True)
#     model = netBuild.attach_relu_layer(model)
#     model = netBuild.attach_conv_layer(model, 32, summary=True)
#     model = netBuild.attach_relu_layer(model)
#     model = netBuild.attach_pooling_layer(model)
#
#     model = netBuild.attach_conv_layer(model, 64, summary=True)
#     model = netBuild.attach_relu_layer(model)
#     model = netBuild.attach_conv_layer(model, 64, summary=True)
#     model = netBuild.attach_relu_layer(model)
#     model = netBuild.attach_pooling_layer(model)
#
#     model = netBuild.attach_conv_layer(model, 128, summary=True)
#     model = netBuild.attach_relu_layer(model)
#     model = netBuild.attach_conv_layer(model, 128, summary=True)
#     model = netBuild.attach_relu_layer(model)
#     model = netBuild.attach_pooling_layer(model)
#
#     model = netBuild.flatten(model)
#     model = netBuild.attach_dense_layer(model, 200, summary=True)
#     model = netBuild.attach_sigmod_layer(model)
#     model = netBuild.attach_dense_layer(model, 32, summary=True)
#     model = netBuild.attach_sigmod_layer(model)
#     model = netBuild.attach_dense_layer(model, 10)
#
#     prediction = netBuild.attach_soft_max_layer(model)
#
# saver = tf.train.Saver()
# with tf.Session() as sess:
#     saver.restore(sess, tf.train.latest_checkpoint("Classifier/saved-model-v2/"))
#     pre = sess.run([prediction], feed_dict={input_img: img128})




font = cv2.FONT_HERSHEY_SIMPLEX
# capture.set(cv.CV_CAP_PROP_FRAME_WIDTH, 320)
# capture.set(cv.CV_CAP_PROP_FRAME_HEIGHT, 240)
# capture.set(3,1800)
# capture.set(4,1800)
capture = cv2.VideoCapture(0)

flag = True

app = wx.App()
# frame = wx.Frame(None)
frame = MyFrame(None, 'Face Recognition', capture,recognizer,faceCascade)
# video = VideoPanel(frame, capture)
frame.Show()
app.MainLoop()

capture.release()
cv2.destroyAllWindows()
