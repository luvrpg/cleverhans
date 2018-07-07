import argparse
import timeit

import keras
import os
import tensorflow as tf
import numpy as np

from keras.datasets import cifar10

from cleverhans.attacks import FastGradientMethod, DeepFool, LBFGS, SaliencyMapMethod, BasicIterativeMethod
from cleverhans.attacks_tf import jacobian_graph, jacobian_augmentation
from cleverhans.utils import AccuracyReport, TemporaryLogLevel
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.utils_tf import model_eval, model_train, batch_eval
from my_tests.models.net_in_net import NetworkInNetwork
from my_tests.models.resnet import ResNet
from my_tests.models.vgg import cifar10vgg


def prepare_cifar_data(vgg=None, resnet=None, net_in_net=None, densenet=None, train=False):
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
        if train:
            x_train = (x_train - mean) / (std + 1e-7)

    if resnet or net_in_net or densenet:
        if x_test.ndim < 4:
            x_test = np.array([x_test])
        mean = [125.307, 122.95, 113.865]
        std = [62.9932, 62.0887, 66.7048]
        for img in x_test:
            for i in range(3):
                img[:, :, i] = (img[:, :, i] - mean[i]) / std[i]

        if train:
            if x_train.ndim < 4:
                x_train = np.array([x_test])
            for img in x_train:
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


def get_model_wrapper(vgg, resnet, net_in_net, densenet):
    model_wrapper = None
    if vgg:
        model_wrapper = cifar10vgg(train=False)
    if resnet:
        model_wrapper = ResNet()
    if net_in_net:
        model_wrapper = NetworkInNetwork()

    return model_wrapper


def get_adversarial_attack_and_params(attack_name, wrap, sess):
    params = None
    stop_gradient = False

    if attack_name == "fgsm":
        attack = FastGradientMethod(wrap, sess=sess)
        params = {'eps': 0.3,
                  'clip_min': 0.,
                  'clip_max': 1.}
        stop_gradient = True
    if attack_name == "deepfool":
        attack = DeepFool(wrap, sess=sess)
    if attack_name == "lbfgs":
        attack = LBFGS(wrap, sess=sess)
    if attack_name == "saliency":
        attack = SaliencyMapMethod(wrap, sess=sess)
    if attack_name == "bim":
        attack = BasicIterativeMethod(wrap, sess=sess)

    return attack, params, stop_gradient


def train_sub(sess, x, y, bbox_preds, X_sub, Y_sub, nb_classes=10,
              nb_epochs_s=250, batch_size=128, learning_rate=0.001, data_aug=6, lmbda=0.1,
              rng=None):
    """
    This function creates the substitute by alternatively
    augmenting the training data and training the substitute.
    :param sess: TF session
    :param x: input TF placeholder
    :param y: output TF placeholder
    :param bbox_preds: output of black-box model predictions
    :param X_sub: initial substitute training data
    :param Y_sub: initial substitute training labels
    :param nb_classes: number of output classes
    :param nb_epochs_s: number of epochs to train substitute model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :param data_aug: number of times substitute training data is augmented
    :param lmbda: lambda from arxiv.org/abs/1602.02697
    :param rng: numpy.random.RandomState instance
    :return:
    """
    # Define TF model graph (for the black-box model)
    model_wrapper = cifar10vgg(empty_model=True)
    model_sub = model_wrapper.model
    preds_sub = model_sub(x)
    print("Defined TensorFlow model graph for the substitute.")

    # Define the Jacobian symbolically using TensorFlow
    grads = jacobian_graph(preds_sub, x, nb_classes)

    # Train the substitute and augment dataset alternatively
    for rho in range(data_aug):
        print("Substitute training epoch #" + str(rho))
        train_params = {
            'nb_epochs': nb_epochs_s,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        }
        with TemporaryLogLevel(tf.logging.WARNING, "cleverhans.utils.tf"):
            model_train(sess, x, y, preds_sub, X_sub, Y_sub,
                        init_all=False, args=train_params)

        # If we are not at last substitute training iteration, augment dataset
        if rho < data_aug - 1:
            print("Augmenting substitute training data.")
            # Perform the Jacobian augmentation
            lmbda_coef = 2 * int(int(rho / 3) != 0) - 1
            X_sub = jacobian_augmentation(sess, x, X_sub, Y_sub, grads,
                                          lmbda_coef * lmbda)

            print("Labeling substitute training data.")
            # Label the newly generated synthetic points using the black-box
            Y_sub = np.hstack([Y_sub, Y_sub])
            X_sub_prev = X_sub[int(len(X_sub)/2):]
            eval_params = {'batch_size': batch_size}
            bbox_val = batch_eval(sess, [x], [bbox_preds], [X_sub_prev], args=eval_params)[0]
            # Note here that we take the argmax because the adversary
            # only has access to the label (not the probabilities) output
            # by the black-box model
            Y_sub[int(len(X_sub)/2):] = np.argmax(bbox_val, axis=1)

    return model_sub, preds_sub


def run(vgg=False, resnet=False, net_in_net=False, densenet=False, attack_name=None, train=True,
        holdout=150):
    start = timeit.default_timer()
    keras.layers.core.K.set_learning_phase(0)
    sess = tf.Session()
    keras.backend.set_session(sess)

    model_wrapper = get_model_wrapper(vgg, resnet, net_in_net, densenet)
    if model_wrapper is None:
        Exception("No model provided")

    model = model_wrapper.model

    x_train, x_test, y_train, y_test = prepare_cifar_data(vgg=vgg, resnet=resnet, net_in_net=net_in_net,
                                                          densenet=densenet, train=train)

    # Initialize substitute training set reserved for adversary
    x_sub = x_test[:holdout]
    y_sub = np.argmax(y_test[:holdout], axis=1)

    # Redefine test set as remaining samples unavailable to adversaries
    x_test = x_test[holdout:]
    y_test = y_test[holdout:]

    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    bbox_preds = model(x)

    print("Training the substitute model.")
    train_sub_out = train_sub(sess, x, y, bbox_preds, x_sub, y_sub)
    model_sub, preds_sub = train_sub_out

    # Evaluate the substitute model on clean test examples
    eval_params = {'batch_size': 128}
    acc = model_eval(sess, x, y, preds_sub, x_test, y_test, args=eval_params)
    print('Test accuracy of substitute on legitimate examples ' + str(acc))

    wrap = KerasModelWrapper(model_sub)
    attack, params, stop_gradient = get_adversarial_attack_and_params(attack_name, wrap, sess)
    params = {"y_target": y, "batch_size": 1}

    adv_x = attack.generate(x, **params) if params else attack.generate(x)
    if stop_gradient:
        # Consider the attack to be constant
        adv_x = tf.stop_gradient(adv_x)

    # Evaluate the accuracy of the "black-box" model on adversarial examples
    accuracy = model_eval(sess, x, y, model(adv_x), x_test, y_test, args=eval_params)
    print('Test accuracy of oracle on adversarial examples generated '
          'using the substitute: ' + str(accuracy))

    stop = timeit.default_timer()
    print(str(stop - start))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    parser = argparse.ArgumentParser(description='Attack models on Cifar10')
    parser.add_argument('--model', default=None)
    parser.add_argument('--attack', default="fgsm")
    parser.add_argument('--train', default=True)

    args = parser.parse_args()
    model_to_run = args.model
    attack_to_run = args.attack
    do_train = args.train

    is_vgg = True if model_to_run == "vgg" else False
    is_resnet = True if model_to_run == "resnet" else False
    is_net_in_net = True if model_to_run == "net_in_net" else False
    is_densenet = True if model_to_run == "densenet" else False

    run(vgg=is_vgg, resnet=is_resnet, net_in_net=is_net_in_net, densenet=is_densenet, attack_name=attack_to_run, train=do_train)
