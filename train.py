import os
import cv2
import glob
import yaml

from math import floor
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.models import load_model
import numpy as np

#Quick Summary of Code - AJ Bamgbelu
#This takes the images that you have saved in the data directory (images should be in their respective categroy within an expirement), and it cuts them into chunks and the runs a classifier model on them
#This code is compatiable with as many categories that you create within an experiment.



#This extracts the chunks in the image
def extract_chunks(images, chunk_size):
    #Needs at least one image in the folder to run
    height, width, _ = images[0].shape
    num_rows = height // chunk_size
    num_cols = width // chunk_size
    arr = [] #This is the array that returns the chunks all of the images for one particular category

    #This is an outer loop so that it traverses through each image in the array of images
    for image in images:
        for row in range(num_rows):
            for col in range(num_cols):
                y_start = row * chunk_size
                y_end = y_start + chunk_size
                x_start = col * chunk_size
                x_end = x_start + chunk_size

                sub_image = image[y_start:y_end, x_start:x_end]

                # see if sub_image is invalid or contains any white pixels
                if not sub_image.any():
                    break
                check_wp = cv2.cvtColor(sub_image, cv2.COLOR_BGR2GRAY)
                if 255 in check_wp or not (check_wp.shape[0] == chunk_size and check_wp.shape[1] == chunk_size):
                    continue
                
                #print(sub_image.shape)
                arr.append(sub_image) #This adds the image chunk to the array of chunks
    return arr

def get_the_last_directory(path):
    path = path.rstrip('/')
    return path.split('/')[-1]
                
def traverse_experiments():
    filepath = "/home/{}/Documents/data/*"

    for experiment in glob.glob(filepath.format(os.environ['USER'])):
        #print(experiment)
        experimentString = experiment + "/*"
        classifierArr = [] #For every experiment there is an array with N cells: One for A and one for B (in order to distinct what im classifying between)
        for category in glob.glob(experimentString):
            #print(category)
            if(category == (experiment + "/config.yaml")):
                continue

            imageArr = [] #This is an array of images for each category
            categoryString = category + "/*"
            for image in glob.glob(categoryString):
                print(image)
                imageArr.append(cv2.imread(image)) #This is appending each image of a category ('pool', 'reef', etc) to one array


            config = yaml.safe_load(open(experiment + "/config.yaml"))
            chunk_size = config['chunk_size']
            extracted_chunks = extract_chunks(imageArr, chunk_size) # This array now holds the image chunks that will be used to train the model
            classifierArr.append(extracted_chunks) #This now has 1/N or 2/N or N/N cells one for each category (N) of the expirement

        print(len(classifierArr))
        path_name = get_the_last_directory(experiment)
        train(classifierArr, path_name, len(classifierArr)) #The classifierArr can now be sent to the training method where it is loaded and then trained

def train(classifierArr, path_name, num_categories):
    #assert len(classifierArr) == 2
    (x_train, y_train), (x_test, y_test) = configure_test_and_train_sets(classifierArr)
    x_train = (x_train / 255)
    x_test = (x_test / 255)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    model = Sequential() 

    model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(20,20,3)))
    model.add(Conv2D(32, kernel_size=3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(num_categories, activation="softmax"))

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 30)

    model_name = path_name + "_cnn.h5"
    model.save(model_name)
        
def configure_test_and_train_sets(cutImages):
    x_train = [] # shape should be (x,20,20, 3)
    y_train = []

    x_test = []
    y_test = []

    #generalizable classification laoading
    y_increment = 0
    for category in cutImages:
        
        for imageIndex in range(len(category)//2):
            x_train.append(category[imageIndex])
            y_train.append(y_increment)
        
        for imageIndex in range(len(category)//2, len(category)):
            x_test.append(category[imageIndex])
            y_test.append(y_increment)

        y_increment += 1


    # [ 0 ] * 200
    # == [0, 0, ..., 0]
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    return (x_train, y_train), (x_test, y_test)

traverse_experiments()

def main():
    pass

if __name__ == "__main__":
    main()
