
######################################################################################
############ IMPORT #######################################################
######################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
import logging

from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils import AccuracyReport, set_log_level

import os

FLAGS = flags.FLAGS

from six.moves import xrange
from cleverhans.utils import to_categorical
from cleverhans.utils import set_log_level
from cleverhans.utils_tf import model_train, model_eval, batch_eval
from cleverhans.attacks_tf import jacobian_graph, jacobian_augmentation

from cleverhans_tutorials.tutorial_models import make_basic_cnn, MLP
from cleverhans_tutorials.tutorial_models import Flatten, Linear, ReLU, Softmax
from cleverhans.utils import TemporaryLogLevel


######################################################################################
############ CROSS    TRAINING FUNCTION #######################################################
######################################################################################

def cross_training(x,y,X_train, Y_train, X_test, Y_test,sess,train_start=0, train_end=60000, test_start=0,
                   test_end=10000, nb_epochs=10, batch_size=128,
                   learning_rate=0.001,
                   clean_train=True,
                   testing=False,
                   backprop_through_attack=False,
                   nb_filters=64, num_threads=None,circles=5,rng=None):
    """
    
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :param clean_train: perform normal training on clean examples only
                        before performing adversarial training.
    :param testing: if true, complete an AccuracyReport for unit tests
                    to verify that performance is adequate
    :param backprop_through_attack: If True, backprop through adversarial
                                    example construction process during
                                    adversarial training.
    :param clean_train: if true, train on clean examples
    :return: an AccuracyReport object
    """

    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # Set TF random seed to improve reproducibility
    # tf.set_random_seed(1234)

    # Set logging level to see debug information
    set_log_level(logging.DEBUG)

    # Create TF session
    # if num_threads:
    #     config_args = dict(intra_op_parallelism_threads=1)
    # else:
    #     config_args = {}
    # sess = tf.Session(config=tf.ConfigProto(**config_args))

    # Get MNIST test data
    # X_train, Y_train, X_test, Y_test = data_mnist(train_start=train_start,
    #                                               train_end=train_end,
    #                                               test_start=test_start,
    #                                               test_end=test_end)

    # Use label smoothing
    assert Y_train.shape[1] == 10
    label_smooth = .1
    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)

    # Define input TF placeholder
    # x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    # y = tf.placeholder(tf.float32, shape=(None, 10))

    model_path = "models/mnist"
    # Train an MNIST model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }
    fgsm_params = {'eps': 0.3,
                   'clip_min': 0.,
                   'clip_max': 1.}
    # rng = np.random.RandomState([2017, 8, 30])

    if clean_train:
        model = make_basic_cnn(nb_filters=nb_filters)
        preds = model.get_probs(x)

        def evaluate():
            # Evaluate the accuracy of the MNIST model on legitimate test
            # examples
            eval_params = {'batch_size': batch_size}
            acc = model_eval(
                sess, x, y, preds, X_test, Y_test, args=eval_params)#Are  X_test&Y_test defined?
            report.clean_train_clean_eval = acc
            assert X_test.shape[0] == test_end - test_start, X_test.shape
            print('Test accuracy on legitimate examples: %0.4f' % acc)
        model_train(sess, x, y, preds, X_train, Y_train, evaluate=evaluate,#model_train should call evaluate?
                    args=train_params, rng=rng)

        # Calculate training error
        if testing:
            eval_params = {'batch_size': batch_size}
            acc = model_eval(
                sess, x, y, preds, X_train, Y_train, args=eval_params)
            report.train_clean_train_clean_eval = acc

        # Initialize the Fast Gradient Sign Method (FGSM) attack object and
        # graph
        fgsm = FastGradientMethod(model, sess=sess)
        adv_x = fgsm.generate(x, **fgsm_params)
        preds_adv = model.get_probs(adv_x)

        # Evaluate the accuracy of the MNIST model on adversarial examples
        eval_par = {'batch_size': batch_size}
        acc = model_eval(sess, x, y, preds_adv, X_test, Y_test, args=eval_par)
        print('Test accuracy on adversarial examples: %0.4f\n' % acc)
        report.clean_train_adv_eval = acc

        # Calculate training error
        if testing:
            eval_par = {'batch_size': batch_size}
            acc = model_eval(sess, x, y, preds_adv, X_train,
                             Y_train, args=eval_par)
            report.train_clean_train_adv_eval = acc

        
    #print("Repeating the process, using adversarial training")
    ############################################################################
    ###################### MAKE #############################################
    ############################################################################
    model_1 = make_basic_cnn(nb_filters=nb_filters)
    model_2 = make_basic_cnn(nb_filters=nb_filters)
    
    # Redefine TF model graph
    for i in range(circles):
        ############################################################################
        ###################### model_1 #############################################
        ############################################################################
        preds_1 = model_1(x)
        fgsm1 = FastGradientMethod(model_2, sess=sess)
        adv_x_1 = fgsm1.generate(x, **fgsm_params)
        if not backprop_through_attack:
            # For the fgsm attack used in this tutorial, the attack has zero
            # gradient so enabling this flag does not change the gradient.
            # For some other attacks, enabling this flag increases the cost of
            # training, but gives the defender the ability to anticipate how
            # the atacker will change their strategy in response to updates to
            # the defender's parameters.
            adv_x_1 = tf.stop_gradient(adv_x_1)
        preds_1_adv = model_1(adv_x_1)

        def evaluate_1():
            # Accuracy of adversarially trained model on legitimate test inputs
            eval_params = {'batch_size': batch_size}
            accuracy = model_eval(sess, x, y, preds_1, X_test, Y_test,
                                  args=eval_params)
            print('Test accuracy on legitimate examples: %0.4f' % accuracy)
            report.adv_train_clean_eval = accuracy

            # Accuracy of the adversarially trained model on adversarial examples
            accuracy = model_eval(sess, x, y, preds_1_adv, X_test,
                                  Y_test, args=eval_params)
            print('Test accuracy on adversarial examples: %0.4f' % accuracy)
            report.adv_train_adv_eval = accuracy

        # Perform and evaluate adversarial training
        model_train(sess, x, y, preds_1, X_train, Y_train,
                    predictions_adv=preds_1_adv, evaluate=evaluate_1,
                    args=train_params, rng=rng)
        ############################################################################
        ###################### model_2 #############################################
        ############################################################################
        model_2 = make_basic_cnn(nb_filters=nb_filters)
        preds_2 = model_2(x)
        fgsm2 = FastGradientMethod(model_1, sess=sess)
        adv_x_2 = fgsm2.generate(x, **fgsm_params)
        if not backprop_through_attack:
            # For the fgsm attack used in this tutorial, the attack has zero
            # gradient so enabling this flag does not change the gradient.
            # For some other attacks, enabling this flag increases the cost of
            # training, but gives the defender the ability to anticipate how
            # the atacker will change their strategy in response to updates to
            # the defender's parameters.
            adv_x_2 = tf.stop_gradient(adv_x_2)
        preds_2_adv = model_2(adv_x_2)

        def evaluate_2():
            # Accuracy of adversarially trained model on legitimate test inputs
            eval_params = {'batch_size': batch_size}
            accuracy = model_eval(sess, x, y, preds_2, X_test, Y_test,
                                  args=eval_params)
            print('Test accuracy on legitimate examples: %0.4f' % accuracy)
            report.adv_train_clean_eval = accuracy

            # Accuracy of the adversarially trained model on adversarial examples
            accuracy = model_eval(sess, x, y, preds_2_adv, X_test,
                                  Y_test, args=eval_params)
            print('Test accuracy on adversarial examples: %0.4f' % accuracy)
            report.adv_train_adv_eval = accuracy

        # Perform and evaluate adversarial training
        model_train(sess, x, y, preds_2, X_train, Y_train,
                    predictions_adv=preds_2_adv, evaluate=evaluate_2,
                    args=train_params, rng=rng)
######################################################################################
################################## TESTING ###########################################
######################################################################################
    # Calculate training errors
    if testing:
        eval_params = {'batch_size': batch_size}
        accuracy = model_eval(sess, x, y, preds_1, X_train, Y_train,
                              args=eval_params)
        report.train_adv_train_clean_eval = accuracy
        accuracy = model_eval(sess, x, y, preds_1_adv, X_train,
                              Y_train, args=eval_params)
        report.train_adv_train_adv_eval = accuracyeval_params = {'batch_size': batch_size}
        accuracy = model_eval(sess, x, y, preds_2, X_train, Y_train,
                              args=eval_params)
        report.train_adv_train_clean_eval = accuracy
        accuracy = model_eval(sess, x, y, preds_2_adv, X_train,
                              Y_train, args=eval_params)
        report.train_adv_train_adv_eval = accuracy

    return model_1,model_2,preds_1,preds_2  #end of cross training

######################################################################################
############ MAIN FUNCTION #######################################################
######################################################################################

def models(x,y,sess,X_train, Y_train, X_test, Y_test,rng):
    m1,m2,p1,p2=cross_training(x,y,X_train, Y_train, X_test, Y_test,sess,nb_epochs=FLAGS.nb_epochs1, batch_size=FLAGS.batch_size1,
                   learning_rate=FLAGS.learning_rate1,
                   clean_train=FLAGS.clean_train1,
                   backprop_through_attack=FLAGS.backprop_through_attack1,
                   nb_filters=FLAGS.nb_filters1,
                   circles=FLAGS.circles,rng=rng)
    
    
    return m1,m2,p1,p2

######################################################################################
############ BLACKBOX TESTING #######################################################
######################################################################################
def setup():
    """
    Helper function to check correct configuration of tf for tutorial
    :return: True if setup checks completed
    """

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    return True


def prep_bbox(sess, x, y, X_train, Y_train, X_test, Y_test,
              nb_epochs, batch_size, learning_rate,
              rng):
    """
    Define and train a model that simulates the "remote"
    black-box oracle described in the original paper.
    :param sess: the TF session
    :param x: the input placeholder for MNIST
    :param y: the ouput placeholder for MNIST
    :param X_train: the training data for the oracle
    :param Y_train: the training labels for the oracle
    :param X_test: the testing data for the oracle
    :param Y_test: the testing labels for the oracle
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param rng: numpy.random.RandomState
    :return:
    """

    # Define TF model graph (for the black-box model)
    model = make_basic_cnn()
    predictions = model(x)
    print("Defined TensorFlow model graph.")

    # Train an MNIST model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }
    model_train(sess, x, y, predictions, X_train, Y_train,
                args=train_params, rng=rng)

    # Print out the accuracy on legitimate data
    eval_params = {'batch_size': batch_size}
    accuracy = model_eval(sess, x, y, predictions, X_test, Y_test,
                          args=eval_params)
    print('Test accuracy of black-box on legitimate test '
          'examples: ' + str(accuracy))

    return model, predictions, accuracy


def substitute_model(img_rows=28, img_cols=28, nb_classes=10):
    """
    Defines the model architecture to be used by the substitute. Use
    the example model interface.
    :param img_rows: number of rows in input
    :param img_cols: number of columns in input
    :param nb_classes: number of classes in output
    :return: tensorflow model
    """
    input_shape = (None, img_rows, img_cols, 1)

    # Define a fully connected model (it's different than the black-box)
    layers = [Flatten(),
              Linear(200),
              ReLU(),
              Linear(200),
              ReLU(),
              Linear(nb_classes),
              Softmax()]

    return MLP(layers, input_shape)


def train_sub(sess, x, y, bbox_preds, X_sub, Y_sub, nb_classes,
              nb_epochs_s, batch_size, learning_rate, data_aug, lmbda,
              rng):
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
    model_sub = substitute_model()
    preds_sub = model_sub(x)
    print("Defined TensorFlow model graph for the substitute.")

    # Define the Jacobian symbolically using TensorFlow
    grads = jacobian_graph(preds_sub, x, nb_classes)

    # Train the substitute and augment dataset alternatively
    for rho in xrange(data_aug):
        print("Substitute training epoch #" + str(rho))
        train_params = {
            'nb_epochs': nb_epochs_s,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        }
        with TemporaryLogLevel(logging.WARNING, "cleverhans.utils.tf"):
            model_train(sess, x, y, preds_sub, X_sub,
                        to_categorical(Y_sub, nb_classes),
                        init_all=False, args=train_params, rng=rng)

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
            bbox_val = batch_eval(sess, [x], [bbox_preds], [X_sub_prev],
                                  args=eval_params)[0]
            # Note here that we take the argmax because the adversary
            # only has access to the label (not the probabilities) output
            # by the black-box model
            Y_sub[int(len(X_sub)/2):] = np.argmax(bbox_val, axis=1)

    return model_sub, preds_sub


def mnist_blackbox(train_start=0, train_end=60000, test_start=0,
                   test_end=10000, nb_classes=10, batch_size=128,
                   learning_rate=0.001, nb_epochs=10, holdout=150, data_aug=6,
                   nb_epochs_s=10, lmbda=0.1):
    """
    MNIST tutorial for the black-box attack from arxiv.org/abs/1602.02697
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :return: a dictionary with:
             * black-box model accuracy on test set
             * substitute model accuracy on test set
             * black-box model accuracy on adversarial examples transferred
               from the substitute model
    """

    # Set logging level to see debug information
    set_log_level(logging.DEBUG)

    # Dictionary used to keep track and return key accuracies
    accuracies = {}

    # Perform tutorial setup
    assert setup()

    # Create TF session
    sess = tf.Session()

    # Get MNIST data
    X_train, Y_train, X_test, Y_test = data_mnist(train_start=train_start,
                                                  train_end=train_end,
                                                  test_start=test_start,
                                                  test_end=test_end)

    # Initialize substitute training set reserved for adversary
    X_sub = X_test[:holdout]
    Y_sub = np.argmax(Y_test[:holdout], axis=1)

    # Redefine test set as remaining samples unavailable to adversaries
    X_test = X_test[holdout:]
    Y_test = Y_test[holdout:]

    # Define input and output TF placeholders
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    # Seed random number generator so tutorial is reproducible
    rng = np.random.RandomState([2017, 8, 30])

    # Simulate the black-box model locally
    # You could replace this by a remote labeling API for instance
    print("Preparing the black-box model.")
    # prep_bbox_out = prep_bbox(sess, x, y, X_train, Y_train, X_test, Y_test,
    #                           nb_epochs, batch_size, learning_rate,
    #                           rng=rng)
    # model, bbox_preds, accuracies['bbox'] = prep_bbox_out
    #crosstraining models:
    m1,m2,p1,p2=models(x,y,sess,X_train, Y_train, X_test, Y_test,rng):
    # Train substitute using method from https://arxiv.org/abs/1602.02697
    print("Training the substitute model.")
    train_sub_out = train_sub(sess, x, y, p1, X_sub, Y_sub,
                              nb_classes, nb_epochs_s, batch_size,
                              learning_rate, data_aug, lmbda, rng=rng)
    model_sub, preds_sub = train_sub_out

    # Evaluate the substitute model on clean test examples
    eval_params = {'batch_size': batch_size}
    acc = model_eval(sess, x, y, preds_sub, X_test, Y_test, args=eval_params)

    accuracies['sub'] = acc

    # Initialize the Fast Gradient Sign Method (FGSM) attack object.
    fgsm_par = {'eps': 0.3, 'ord': np.inf, 'clip_min': 0., 'clip_max': 1.}
    fgsm = FastGradientMethod(model_sub, sess=sess)

    # Craft adversarial examples using the substitute
    eval_params = {'batch_size': batch_size}
    x_adv_sub = fgsm.generate(x, **fgsm_par)

    # Evaluate the accuracy of the "black-box" model on adversarial examples
    accuracy = model_eval(sess, x, y, m1(x_adv_sub), X_test, Y_test,
                          args=eval_params)
    print('Test accuracy of oracle on adversarial examples generated '
          'using the substitute: ' + str(accuracy))
    accuracies['bbox_on_sub_adv_ex'] = accuracy

    return accuracies

######################################################################################
############ MAIN #######################################################
######################################################################################
def main(argv=None):
    mnist_blackbox(nb_classes=FLAGS.nb_classes, batch_size=FLAGS.batch_size,
                   learning_rate=FLAGS.learning_rate,
                   nb_epochs=FLAGS.nb_epochs, holdout=FLAGS.holdout,
                   data_aug=FLAGS.data_aug, nb_epochs_s=FLAGS.nb_epochs_s,
                   lmbda=FLAGS.lmbda)


######################################################################################
############ CONSTANTS DEFINATION #######################################################
######################################################################################


if __name__ == '__main__':
    flags.DEFINE_integer('nb_filters1', 64, 'Model size multiplier')
    flags.DEFINE_integer('nb_epochs1', 6, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size1', 128, 'Size of training batches')
    flags.DEFINE_integer('circles',5,'Circles of cross_training')
    flags.DEFINE_float('learning_rate1', 0.001, 'Learning rate for training')
    flags.DEFINE_bool('clean_train1', False, 'Train on clean examples')#No clean_train 
    flags.DEFINE_bool('backprop_through_attack1', False,
                      ('If True, backprop through adversarial example '
                       'construction process during adversarial training'))
    flags.DEFINE_integer('nb_classes', 10, 'Number of classes in problem')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')

    # Flags related to oracle
    flags.DEFINE_integer('nb_epochs', 10, 'Number of epochs to train model')

    # Flags related to substitute
    flags.DEFINE_integer('holdout', 150, 'Test set holdout for adversary')
    flags.DEFINE_integer('data_aug', 6, 'Nb of substitute data augmentations')
    flags.DEFINE_integer('nb_epochs_s', 10, 'Training epochs for substitute')
    flags.DEFINE_float('lmbda', 0.1, 'Lambda from arxiv.org/abs/1602.02697')

    tf.app.run()
    
