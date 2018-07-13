import glob
from random import shuffle

import cv2
import os

import keras
import numpy as np
import tensorflow as tf

from cleverhans.utils import AccuracyReport
from my_tests.cifar10_whitebox import eval_model_accuracy
from my_tests.models.resnet import ResNet


def train_detector():
    detector = ResNet(load_weights=False, det=True)
    detector.train2()


def eval_detector():
    sess = tf.Session()
    keras.backend.set_session(sess)

    resnet = ResNet(load_weights=False, det=True)
    report = AccuracyReport()

    all_imgs = []
    query = os.path.join("/home/natalia/test/", "**", "*.jpg")
    all_imgs.extend(glob.glob(query, recursive=True))

    clean_set = []
    adv_set = []
    for i in all_imgs:
        img = cv2.imread(i)
        if "adv" in i:
            adv_set.append(img)
            continue
        clean_set.append(img)

    y_clean = [[0] for _ in range(len(clean_set))]
    y_adv = [[1] for _ in range(len(adv_set))]

    shuffle(clean_set)
    shuffle(adv_set)
    x_test = adv_set + clean_set
    y_test = y_adv + y_clean

    y_test = keras.utils.to_categorical(y_test, 2)

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    y = tf.placeholder(tf.float32, shape=(None, 2))
    predictions = resnet.model(x)

    eval_model_accuracy(sess, x, y, predictions, x_test, y_test, report)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    eval_detector()
