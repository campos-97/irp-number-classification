from tkinter import *
from tkinter.colorchooser import askcolor
import pyscreenshot as ImageGrab
from PIL import Image
import numpy as np
import threading
import matplotlib.pyplot as plt
import time
import datetime as dt
import joblib
import os

#keras
from keras.datasets.mnist import load_data
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
#fetch original mnist dataset
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle           # To shuffle vectors

import mnist
# import custom module
from mnist_helpers import *

class Paint(object):

    DEFAULT_PEN_SIZE = 5.0
    DEFAULT_COLOR = 'black'

    def __init__(self):
        self.root = Tk()

        #data field is 70k x 784 array, each row represents pixels from 28x28=784 image
        self.images,self.targets = self.load_images_targets()

        #full dataset classification
        self.X_data = self.images/255.0
        self.Y = self.targets

        #split data to train and test
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data(self.images, self.targets)

        self.modelsOnTraining = 0
        self.kernel = StringVar()
        self.shrinking = IntVar()
        self.probability = IntVar()
        self.stoppingCriterion = StringVar()
        self.chacheSize = StringVar()
        self.maxIter = StringVar()
        self.degree = StringVar()
        self.gamma  = StringVar()
        self.coef   = StringVar()
        self.C = StringVar()
        self.verbose = IntVar()
        self.guess = StringVar()
        self.usingKeras = IntVar()

        # Default values
        self.kernels = {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'};
        self.kernel.set('rbf')
        self.shrinking.set(1)
        self.probability.set(0)
        self.stoppingCriterion.set('0.001')
        self.chacheSize.set('200')
        self.maxIter.set('100')
        self.degree.set('3')
        self.gamma.set('auto')
        self.coef.set('0.0')
        self.C.set('5')
        self.verbose.set(0)
        self.guess.set('0')
        self.usingKeras.set(0)

        self.models = {'None'}
        self.model = None
        self.modelName = StringVar()
        # link function to change model
        self.modelName.trace('w', self.selectModel)
        self.getModels()


        self.eraser_button = Button(self.root, text='eraser', command=self.use_eraser)
        self.eraser_button.grid(row=0, column=3)

        self.clear_button = Button(self.root, text='clear', command=self.clearCanvas)
        self.clear_button.grid(row=0, column=2)

        self.classify = Button(self.root, text='classify', command=self.classify)
        self.classify.grid(row=0, column=0)

        self.choose_size_button = Scale(self.root, from_=10, to=50, orient=HORIZONTAL)
        self.choose_size_button.grid(row=0, column=4)

        self.c = Canvas(self.root, bg='white', width=600, height=600)
        self.c.grid(row=1, columnspan=5)

        self.modelMenu = OptionMenu(self.root, self.modelName, *self.models)
        self.modelMenu.grid(row=0, column=1)
        self.modelMenu.config(width=10)

        self.guess_lbl = Label(self.root, textvariable=self.guess, font=("Helvetica", 72))

        self.train = Button(self.root, text='train', command=self.train)
        self.train.grid(row=7, column=0)

        OptionMenu(self.root, self.kernel, *self.kernels).grid(row=2, column=1)
        Label(text="Choose Kernel").grid(row=2, column=0)

        Checkbutton(self.root, variable=self.shrinking, text="Shrinking").grid(row=2, column=2)
        Checkbutton(self.root, variable=self.probability, text="Probability").grid(row=2, column=3)

        Label(text="StoppingCriterion").grid(row=3, column=0)
        Entry(self.root, textvariable=self.stoppingCriterion, width=10).grid(row=3, column=1)
        Label(text="CacheSize").grid(row=3, column=2)
        Entry(self.root, textvariable=self.chacheSize, width=10).grid(row=3, column=3)
        Label(text="MaxIter").grid(row=4, column=0)
        Entry(self.root, textvariable=self.maxIter, width=10).grid(row=4, column=1)
        Label(text="Degree").grid(row=4, column=2)
        Entry(self.root, textvariable=self.degree, width=10).grid(row=4, column=3)
        Label(text="Gamma").grid(row=5, column=0)
        Entry(self.root, textvariable=self.gamma, width=10).grid(row=5, column=1)
        Label(text="Coef").grid(row=5, column=2)
        Entry(self.root, textvariable=self.coef, width=10).grid(row=5, column=3)
        Label(text="C").grid(row=6, column=0)
        Entry(self.root, textvariable=self.C, width=10).grid(row=6, column=1)
        Checkbutton(self.root, variable=self.verbose, text="Verbose").grid(row=6, column=2)
        
        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = self.choose_size_button.get()
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)    

    def use_eraser(self):
        self.guess_lbl.grid_forget()
        self.eraser_on = not self.eraser_on
        if self.eraser_button.config('relief')[-1] == 'sunken':
            self.eraser_button.config(relief="raised")
        else:
            self.eraser_button.config(relief="sunken")

    def clearCanvas(self):
        self.guess_lbl.grid_forget()
        self.c.delete("all")

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def paint(self, event):
        self.guess_lbl.grid_forget()
        self.line_width = self.choose_size_button.get()
        paint_color = 'white' if self.eraser_on else self.color
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y

    def get_content_image(self):
        x=self.root.winfo_rootx()+self.c.winfo_x()
        y=self.root.winfo_rooty()+self.c.winfo_y()
        x1=x+self.c.winfo_width()
        y1=y+self.c.winfo_height()
        im=ImageGrab.grab((x,y,x1,y1), childprocess=False)
        im=im.resize((28,28),Image.ANTIALIAS)
        im=im.convert('L')
        im=np.array(list(im.getdata()))
        im=1-im/255.0

        if (self.usingKeras.get()):
            return np.reshape(list(im), (1, 28, 28, 1))
        return list(im.reshape(1, -1))


    def classify(self):
        if(self.model != None):
            image = self.get_content_image()
            #list(image.reshape(1, -1))
            prediction = self.model.predict(image)
            if (self.usingKeras.get()):
                prediction = np.argmax(prediction, axis=1)
            self.guess.set(prediction)
            self.guess_lbl.grid(row=1, columnspan=5)

    def train(self):
        self.modelsOnTraining += 1
        self.train.configure(state=DISABLED)
        threading.Thread(target=self.train_thread).start()

    def reset(self, event):
        self.old_x, self.old_y = None, None

    def getModels(self):
        self.models = os.listdir("./models/")
        if(len(self.models) != 0):
            self.modelName.set(self.models[0])

    def selectModel(self, *args):
        if(self.modelName.get() != "None"):
            if(self.modelName.get().split('.')[-1] == "sav"):
                self.usingKeras.set(0)
                self.model = joblib.load("./models/"+self.modelName.get())
            elif(self.modelName.get().split('.')[-1] == "h5"):
                self.usingKeras.set(1)
                del self.model
                self.model = load_model("./models/"+self.modelName.get())
            else:
                print("Not supported format")

    def train_thread(self):
        classifier = svm.SVC(C=float(self.C.get()), kernel=self.kernel.get(), degree=int(self.degree.get()),
         gamma=self.gamma.get(), coef0=float(self.coef.get()), shrinking=self.shrinking.get(),
          probability=self.probability.get(), tol=float(self.stoppingCriterion.get()),
          cache_size=int(self.chacheSize.get()), max_iter=int(self.maxIter.get()), verbose=self.verbose.get())
        #We learn the digits on train part
        classifier.fit(self.X_train, self.y_train)
        self.modelsOnTraining -= 1
        name = self.get_model_name()
        joblib.dump(classifier, "./models/" + name + '.sav')
        print('Saved model' + "./models/" + name + '.sav')
        self.train.configure(state=NORMAL)


    def get_model_name(self):
        name = 'kernel:'+self.kernel.get()
        if(self.kernel.get()=='poly'):
            name += '_degree:'+self.degree.get()
        if(self.kernel.get() in 'rbf-poly-sigmoid'):
            name += '_gamma:'+self.gamma.get()
        if(self.kernel.get() in 'poly-sigmoid'):
            name += '_coef0:'+self.coef.get()
        name += '_chacheSize:'+self.chacheSize.get()
        name += '_C:'+self.C.get()
        if(self.shrinking.get()):
            name += '_shrinking'
        if(self.probability.get()):
            name += '_probability'
        name += '_stoppingCriterion:'+self.stoppingCriterion.get()
        name += '_maxIter:'+self.maxIter.get()
        name += '_'+ str(time.time())
        return name 

    def load_images_targets(self):
        train_images = mnist.train_images()
        train_labels = mnist.train_labels()
        test_images = mnist.test_images()
        test_labels = mnist.test_labels()

        train_images = train_images.reshape((-1, 784))
        test_images = test_images.reshape((-1, 784))

        images = np.concatenate((train_images, test_images),axis=0)
        targets = np.concatenate((train_labels,test_labels),axis=0)

        return images,targets


    def split_data(self, images, targets, param_test_size=10000/70000, param_random_state=42):
        X_data = images/255.0                   #Normalize the data
        Y = targets
        #Split data to train and test
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_data, Y, test_size=param_test_size, random_state=param_random_state, shuffle=False)
        #Now shuffle that data
        X_train, y_train = shuffle(X_train, y_train, random_state=0)
        X_test, y_test = shuffle(X_test, y_test, random_state=0)

        return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    Paint()