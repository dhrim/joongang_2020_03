# 교육 목표

딥러닝의 개념을 이해하고 Keras로 실제 딥러닝 코드를 구현하고 실행시킬 수 있다.


# 대상

python으로 기본적인 프로그래밍을 할 수있는 학생.


# 교육 상세 표표

다음의 코드를 다룰 수 있는 것을 목표로 함
```
# copy from https://www.tensorflow.org/beta/tutorials/keras/basic_classification

!pip install -q tensorflow==2.0.0-beta1

from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)



fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


print(train_images.shape)
print(len(train_labels))
print(train_labels)
print(test_images.shape)
print(len(test_labels))




plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()


train_images = train_images / 255.0
test_images = test_images / 255.0


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()




model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)




test_loss, test_acc = model.evaluate(test_images, test_labels)
print('\n테스트 정확도:', test_acc)

predictions = model.predict(test_images)
print(predictions[0])

```


# 지식 구조

```
Environment
    jupyter
	colab
	usage
		!, %, run
    GCP notebook
linux
	ENV
	command
		cd, pwd, ls
		mkdir, rm, cp
		head, more, tail, cat
	util
		apt
		git, wget
		grep, wc, tree
		tar, unrar, unzip
	gpu
		nvidia-smi

python
	env
		python
			interactive
			execute file
		pip
	syntax
        variable
        data
            tuple
            list
            dict
            set
        loop
        if
        comprehensive list
        function
        class
	module
		import

libray
    numpy
        load
        op
        shape
        slicing
        reshape
        axis + sum, mean
    pandas
        load
        view
        to numpy
    matplot
        draw line graph
        scatter
        show image

Deep Learning
    DNN
        concept
            layer, node, weight, bias, activation
            cost function
            GD, BP
        data
            x, y
            train, validate, test
            shuffle
        learning curve : accuracy, loss
        tunning
            overfitting, underfitting
            regularization, dropout, batch normalization
            data augmentation
        Transfer Learning
    type
        supervised
        unsupervised
        reinforcement
    model
        CNN
            varnilla, VGG16
        RNN
        GAN
    task
        Classification
        Object Detection
        Generation
        target : text/image

TensorFlow/Keras
    basic frame
        data preparing
            x, y
            train, valid, test
            normalization
            ImageDataGenerator
        fit
        evaluate
        predict
    model
        activation function
        initializer
    tuning
        learning rate
        regularier
        dropout
        batch normalization
    save/load
    compile
        optimizer
        loss
        metric
```


# 일자별 계획

## 1일차

실습 환경. numpy

- 교육 환경 : [env.md](material/env.md)
- numpy : 데이터 로딩, 보기, 데이터 변환, 형태 변경 : [library.md](material/library.md)
- linux 기본 명령어 : [linux.md](material/linux.md)
    - bash, cd, ls, rm, mkdir, mv, tar, unzip
    - docker, pip, apt, wget, EVN, git
    

<br>

## 2일차

pandas, matplot

- pandas : 데이터 로딩, 보기, 데이터 추출
- matplot : 그래프 그리기, 이미지 표시하기
- 교육 자료 : [ibrary.md](material/library.md)


<br>

## 3일차

딥러닝, DNN

- 딥러닝 입문
    - 딥러닝 개념 : [deep_learning_intro.pptx](material/deep_learning/deep_learning_intro.pptx)
    - 알파고 이해하기 : [understanding_ahphago.pptx](material/deep_learning/understanding_ahphago.pptx)
- Keras 요약 [keras_in_short.md](material/deep_learning/keras_in_short.md)
- DNN in Keras : [dnn_in_keras_shortly.ipynb](material/deep_learning/dnn_in_keras.ipynb)

<br>

## 4일차

분류기로써의 DNN
- 분류기 : [dnn_as_a_classifier.ipynb](material/deep_learning/dnn_as_a_classifier.ipynb)
- IRIS : [dnn_iris_classification.ipynb](material/deep_learning/dnn_iris_classification.ipynb)
- MNIST 영상데이터 : [dnn_mnist.ipynb](material/deep_learning/dnn_mnist.ipynb)

CNN
- MNIST : [cnn_mnist.ipynb](material/deep_learning/cnn_mnist.ipynb)
- CIFAR10 : [cnn_cifar10.ipynb](material/deep_learning/cnn_cifar10.ipynb)
- VGG - CIFAR10, ImageNet, custom data : [VGG16_classification_and_cumtom_data_training.ipynb](material/deep_learning/VGG16_classification_and_cumtom_data_training.ipynb)


## 5일차

Object Detection
- YOLO : [object_detection.md](material/object_detection.md)



## 12일차

실프로젝트 - OCR
- ASTER
- 교육 자료 : [실무과제_ocr.md](material/실무과제_ocr.md)

## 13일차

실프로제트 - Pose Estimation
- OpenPose, Colab, Webcam
- 교육자료 : [실무과제_pose_estimation.md](material/실무과제_pose_estimation.md)

실프로젝트 : RealTime Object Detection
- YOLO, Colab, WebCam
- 교육자료 : [실무과제_realtime_yolo.md](material/실무과제_realtime_yolo.md)

실프로젝트 : 주가 예측
- DNN
- 교육자료 : [실무과제_stock_prediction.md](material/실무과제_stock_prediction.md)