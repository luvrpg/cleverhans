import argparse

import keras
import os
import tensorflow as tf
import numpy as np

from keras.datasets import cifar10

from cleverhans.attacks import FastGradientMethod, DeepFool
from cleverhans.utils import AccuracyReport
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.utils_tf import model_eval, model_train
from my_tests.models.vgg import cifar10vgg
from networks.densenet import DenseNet
from networks.network_in_network import NetworkInNetwork
from networks.resnet import ResNet


def prepare_cifar_data(vgg=None, resnet=None, net_in_net=None, densenet=None):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    # prepare data for VGG
    if vgg:
        mean = 120.707
        std = 64.15
        x_test = (x_test - mean) / (std + 1e-7)

    if resnet or net_in_net or densenet:
        if x_test.ndim < 4:
            x_test = np.array([x_test])
        mean = [125.307, 122.95, 113.865]
        std = [62.9932, 62.0887, 66.7048]
        for img in x_test:
            for i in range(3):
                img[:, :, i] = (img[:, :, i] - mean[i]) / std[i]

    return x_train, x_test, y_train, y_test


def eval_model(sess, x, y, predictions, x_test, y_test, batch_size=128):
    eval_params = {'batch_size': batch_size}
    return model_eval(sess, x, y, predictions, x_test, y_test, args=eval_params)


def eval_model_accuracy(sess, x, y, predictions, x_test, y_test, report):
    # accuracy on clean samples, model trained on clean samples
    acc = eval_model(sess, x, y, predictions, x_test, y_test)
    print('Test accuracy on legitimate examples: %0.4f\n' % acc)
    report.clean_train_clean_eval = acc


def eval_model_accuracy_adv_samples(sess, x, y, predictions, x_test, y_test, report):
    # accuracy on adversarial samples, model trained on clean samples
    acc = eval_model(sess, x, y, predictions, x_test, y_test)
    print('Test accuracy on adversarial examples: %0.4f\n' % acc)
    report.clean_train_adv_eval = acc


def eval_adv_model(sess, x, y, predictions, predictions_adv, x_test, y_test, report):
    print('MODEL TRAINED WITH ADVERSARIAL EXAMPLES')
    # accuracy on clean samples, model trained on adversarial samples
    acc = eval_model(sess, x, y, predictions, x_test, y_test)
    print('Test accuracy on legitimate examples: %0.4f' % acc)
    report.adv_train_clean_eval = acc

    # accuracy on adversarial examples, model trained on adversarial samples
    acc2 = eval_model(sess, x, y, predictions_adv, x_test, y_test)
    print('Test accuracy on adversarial examples: %0.4f' % acc2)
    report.adv_train_adv_eval = acc2


def run(vgg=False, resnet=False, net_in_net=False, densenet=False):
    keras.layers.core.K.set_learning_phase(0)
    report = AccuracyReport()
    sess = tf.Session()
    keras.backend.set_session(sess)

    model_wrapper = None
    if vgg:
        model_wrapper = cifar10vgg(train=False)
    if resnet:
        model_wrapper = ResNet()
    if net_in_net:
        model_wrapper = NetworkInNetwork()
    if densenet:
        model_wrapper = DenseNet()

    if model_wrapper is None:
        Exception("No model provided")
    model = model_wrapper.model

    x_train, x_test, y_train, y_test = prepare_cifar_data(vgg=vgg, resnet=resnet, net_in_net=net_in_net,
                                                          densenet=densenet)

    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    predictions = model(x)
    print("Defined TensorFlow model graph.")

    eval_model_accuracy(sess, x, y, predictions, x_test, y_test, report)

    # FGSM ATTACK
    wrap = KerasModelWrapper(model)
    fgsm = FastGradientMethod(wrap, sess=sess)
    fgsm_params = {'eps': 0.3,
                   'clip_min': 0.,
                   'clip_max': 1.}
    adv_x = fgsm.generate(x, **fgsm_params)
    # Consider the attack to be constant
    adv_x = tf.stop_gradient(adv_x)
    predictions_adv = model(adv_x)
    eval_model_accuracy_adv_samples(sess, x, y, predictions_adv, x_test, y_test, report)

    # # DEEPFOOL ATTACK
    # deepfool = DeepFool(wrap, sess=sess)
    # adv_x_deepfool = deepfool.generate(x)
    # predictions_adv_deepfool = model(adv_x_deepfool)
    # eval_model_accuracy_adv_samples(sess, x, y, predictions_adv_deepfool, x_test, y_test, report)

    # ADVERSARIAL TRAINING
    vgg_model = cifar10vgg(train=False)
    model_2 = vgg_model.model
    predictions_2 = model_2(x)
    wrap_2 = KerasModelWrapper(model_2)
    fgsm2 = FastGradientMethod(wrap_2, sess=sess)
    predictions_2_adv = model_2(fgsm2.generate(x, **fgsm_params))

    train_params = {
        'nb_epochs': 5,
        'batch_size': 128,
        'learning_rate': 0.001,
        'train_dir': r"C:\Users\pan\Desktop\wyniki\adversarially_crafted_cifar10",
        'filename': "adversarially_crafted.ckpt"
    }
    model_train(sess, x, y, predictions_2, x_train, y_train,
                predictions_adv=predictions_2_adv, evaluate=eval_adv_model,
                args=train_params, save=True)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    parser = argparse.ArgumentParser(description='Attack models on Cifar10')
    parser.add_argument('--model', default=None)
    args = parser.parse_args()
    model = args.model

    is_vgg = True if model == "vgg" else False
    is_resnet = True if model == "resnet" else False
    is_net_in_net = True if model == "net_in_net" else False
    is_densenet = True if model == "densenet" else False

    run(vgg=is_vgg, resnet=is_resnet, net_in_net=is_net_in_net, densenet=is_densenet)
