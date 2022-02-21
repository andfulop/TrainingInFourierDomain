import numpy as np
import random
import tensorflow.compat.v1 as tf
from tensorflow import keras
tf.disable_v2_behavior()
tf.compat.v1.disable_resource_variables()


# Load training and test data (MNISt dataset):
dataLoader = keras.datasets.mnist
(features, labels), (testFeatures, testLabels) = dataLoader.load_data()
onehot_labels = np.zeros((labels.shape[0], 10))
onehot_labels[np.arange(labels.shape[0]), labels] = 1
labels = onehot_labels
features = features
testFeatures = testFeatures
onehot_testLabels = np.zeros((testLabels.shape[0], 10))
onehot_testLabels[np.arange(testLabels.shape[0]), testLabels] = 1
testLabels = onehot_testLabels
# Set the parameters:
NumClasses = 10
BatchLength = 16
Size = [28, 28, 1]
length_of_signal = Size
NumIteration = 40001
LearningRate = 1e-4
EvalFreq = 1000
NumKernels = [16, 32, 64]


# spectral pooling size:
#         1div2  - 0
#         6div8 - 1
specPoolSize = 0


def fourier_complex_relu(x):
    real = tf.real(x)
    imag = tf.imag(x)
    return tf.complex(tf.cast(real*real+imag*imag > 0.1, tf.float32)*real, tf.cast(real*real+imag*imag > 0.1, tf.float32)*imag)


def convolution_in_freq_domain_without_ifft(f_input, out_channels):
    in_shape = f_input.get_shape()
    bias_r = tf.get_variable('BiasReal', [out_channels], dtype=tf.float32)
    bias_c = tf.get_variable('BiasComp', [out_channels], dtype=tf.float32)
    bias = tf.complex(bias_r, bias_c)
    # Spectral pooling:
    if specPoolSize == 0:
        f_input = tf.slice(f_input, [0, int(in_shape[1] // 4), int(in_shape[2] // 4), 0], [-1, int(in_shape[1] // 2), int(in_shape[2] // 2), in_shape[-1]])
    elif specPoolSize == 1:
        f_input = tf.slice(f_input, [0, int(in_shape[1] // 8), int(in_shape[2] // 8), 0], [-1, int(in_shape[1]) - int(2 * in_shape[1] // 8), int(in_shape[2]) - int(2 * in_shape[2] // 8), in_shape[-1]])
    in_shape = f_input.get_shape()
    w_r = tf.get_variable('w_r', [int(in_shape[1]), int(in_shape[2]), int(in_shape[3]), out_channels])
    w_i = tf.get_variable('w_i', [int(in_shape[1]), int(in_shape[2]), int(in_shape[3]), out_channels])
    w = tf.complex(w_r, w_i)
    fourier_kernel = w
    fourier_kernel = tf.tile(tf.expand_dims(fourier_kernel, 0), [BatchLength, 1, 1, 1, 1])
    out = []
    for ind in range(out_channels):
        res = tf.multiply(f_input[:, :, :, :], fourier_kernel[:, :, :, :, ind])
        res = tf.add(res, bias[ind])
        res = tf.expand_dims(tf.reduce_sum(res, 3), -1)
        out.append(res)
    out = tf.concat(out, 3)
    norm_real = tf.layers.batch_normalization(tf.real(out), training=True)
    norm_comp = tf.layers.batch_normalization(tf.imag(out), training=True)
    out = tf.complex(norm_real, norm_comp)
    out = fourier_complex_relu(out)
    return out


tf.reset_default_graph()
InputData = tf.placeholder(tf.float32, [None] + Size)  # input images
OneHotLabels = tf.placeholder(tf.int32, [None, NumClasses])  # the expected outputs, labels


# Take the input to Fourier domain
CurrentInput = tf.cast(InputData, tf.complex64)
CurrentInput = tf.transpose(CurrentInput, [3, 0, 1, 2])
fourierInput = tf.fft2d(CurrentInput, name=None)
fourierInput = tf.transpose(fourierInput, [1, 2, 3, 0])
fourierInput = tf.roll(fourierInput, shift=[int(Size[0]//2), int(Size[1]//2)], axis=[1, 2])
CurrentFilters = Size[-1]
# a loop which creates all layers
for N in range(len(NumKernels)):
    with tf.variable_scope('conv' + str(N)):
        fourierInput = convolution_in_freq_domain_without_ifft(fourierInput, NumKernels[N])
with tf.variable_scope('FC'):
    fourierInput = tf.square(tf.real(fourierInput)) + tf.square(tf.imag(fourierInput))
    CurrentShape = fourierInput.get_shape()
    FeatureLength = int(CurrentShape[1]*CurrentShape[2]*CurrentShape[3])
    FC = tf.reshape(fourierInput, [-1, FeatureLength])
    W = tf.get_variable('W', [FeatureLength, NumClasses])
    FC = tf.matmul(FC, W)
    Bias = tf.get_variable('Bias', [NumClasses])
    FC = tf.add(FC, Bias)


with tf.name_scope('loss'):
    Loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=OneHotLabels, logits=FC))


with tf.name_scope('optimizer'):
    Optimizer = tf.train.AdamOptimizer(LearningRate).minimize(Loss)


with tf.name_scope('accuracy'):
    CorrectPredictions = tf.equal(tf.argmax(FC, 1), tf.argmax(OneHotLabels, 1))
    Accuracy = tf.reduce_mean(tf.cast(CorrectPredictions, tf.float32))


Init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as Sess:
    Sess.run(Init)
    Step = 1
    while Step < NumIteration:
        UsedInBatch = random.sample(range(features.shape[0]), BatchLength)
        batch_xs = features[UsedInBatch, :]
        batch_ys = labels[UsedInBatch, :]
        batch_xs = np.reshape(batch_xs, [BatchLength]+Size)
        _, Acc, L = Sess.run([Optimizer, Accuracy, Loss], feed_dict={InputData: batch_xs, OneHotLabels: batch_ys})
        if (Step % 100) == 0:
            print("Iteration: "+str(Step))
            print("Accuracy:" + str(Acc))
            print("Loss:" + str(L))
        if (Step % EvalFreq) == 0:
            SumAcc = 0.0
            for i in range(0, testFeatures.shape[0]):
                batch_xs = testFeatures[i, :]
                batch_ys = testLabels[i, :]
                batch_xs = np.reshape(batch_xs, [1]+Size)
                batch_ys = np.reshape(batch_ys, [1, NumClasses])
                a = Sess.run(Accuracy, feed_dict={InputData: batch_xs, OneHotLabels: batch_ys})
                SumAcc += a
            print("Independent Test set: "+str(float(SumAcc)/testFeatures.shape[0]))
        Step += 1
