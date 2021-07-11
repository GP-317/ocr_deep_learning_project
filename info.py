# -*- coding: utf8 -*-
import matplotlib.pyplot as plt
import numpy as np


def plot_performance(hist):
    
    plt.style.use("ggplot")
    plt.figure(1)

    plt.plot(hist.history["loss"], label="train_loss") 
    plt.plot(hist.history["val_loss"], label="val_loss") 
    plt.plot(hist.history["accuracy"], label="train_acc") 
    plt.plot(hist.history["val_accuracy"], label="val_acc") 
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    
    plt.show()



def plot_images_sample(X, Y, titre):

    fig = plt.figure(1)
    rand_indicies = np.random.randint(len(X), size=25)
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        index = rand_indicies[i]
        plt.imshow(np.squeeze(X[index]), cmap=plt.cm.binary)
        plt.xlabel(Y[index])
    
    fig.suptitle(titre)
    plt.show()