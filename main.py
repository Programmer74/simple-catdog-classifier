#!/usr/bin/env python3
import os

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from imutils import paths
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score


def extract_histogram(image, bins=(8, 8, 8)):
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def show_image(path):
    plt.figure()
    img = mpimg.imread(path)
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    imagePaths = sorted(list(paths.list_images('./train')))
    data = []
    labels = []

    for (i, imagePath) in enumerate(imagePaths):
        image = cv2.imread(imagePath, 1)
        label = imagePath.split(os.path.sep)[-1].split(".")[0]
        hist = extract_histogram(image)
        data.append(hist)
        labels.append(label)

    le = LabelEncoder()
    labels = le.fit_transform(labels)
    print(labels[0])
    # show_image(imagePaths[0])
    (trainData, testData, trainLabels, testLabels) = train_test_split(np.array(data), labels, test_size=0.25,
                                                                      random_state=5)
    model = LinearSVC(random_state=5, C=0.72)
    model.fit(trainData, trainLabels)
    predictions = model.predict(testData)
    print(classification_report(testLabels, predictions, target_names=le.classes_))
    predictions = model.predict(testData)
    print("F1 score is ", f1_score(testLabels, predictions, average='macro'))
    print("theta 448", model.coef_[0][448])
    print("theta 241", model.coef_[0][241])
    print("theta 75", model.coef_[0][75])

    images_to_predict = ["cat.1046.jpg", "dog.1016.jpg", "cat.1028.jpg", "cat.1032.jpg"]
    for image_to_predict in images_to_predict:
        image_path = "./test/" + image_to_predict
        singleImage = cv2.imread(image_path)
        histt = extract_histogram(singleImage)
        histt2 = histt.reshape(1, -1)
        prediction = model.predict(histt2)
        print("for image", image_path, "prediction is", prediction)