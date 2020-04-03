# 교육 목표

딥러닝의 개념을 이해하고 Keras로 실제 딥러닝 코드를 구현하고 실행시킬 수 있다.


# 대상

python으로 기본적인 프로그래밍을 할 수있는 학생.


# 교육 상세 목표

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

- 딥러닝 개념 : [deep_learning_intro.pptx](material/deep_learning/deep_learning_intro.pptx)
- DNN in Keras : [dnn_in_keras.ipynb](material/deep_learning/dnn_in_keras.ipynb)


<br>

## 2일차

- DNN in Keras : [dnn_in_keras.ipynb](material/deep_learning/dnn_in_keras.ipynb)
- 분류기로서 DNN : [dnn_as_a_classifier.ipynb](material/deep_learning/dnn_as_a_classifier.ipynb)
- DNN IRIS 분류: [dnn_iris_classification.ipynb](material/deep_learning/dnn_iris_classification.ipynb)
- DNN MNIST 분류 : [dnn_mnist.ipynb](material/deep_learning/dnn_mnist.ipynb)


<br>

## 3일차
- CNN 영상분류 - MNIST : [cnn_mnist.ipynb](material/deep_learning/cnn_mnist.ipynb)
- CNN 컬러영상분류 - CIFAR10 : [cnn_cifar10.ipynb](material/deep_learning/cnn_cifar10.ipynb)
- CNN IRIS 분류 : [iris_cnn.ipynb](material/deep_learning/iris_cnn.ipynb)
- VGG로 영상 분류, 전이학습 : [VGG16_classification_and_cumtom_data_training.ipynb](material/deep_learning/VGG16_classification_and_cumtom_data_training.ipynb)
- 알파고 이해하기 : [understanding_ahphago.pptx](material/deep_learning/understanding_ahphago.pptx)

<br>

## 4일차
- text 분류 : [text_classification.ipynb](material/deep_learning/text_classification.ipynb)
- 물체 탐지(Object Detection) 
    - YOLO : [object_detection.md](material/deep_learning/object_detection.md)
    - RetinaNet : [object_detection_retinanet.ipynb](material/deep_learning/object_detection_retinanet.ipynb)
    - annotation tool : https://github.com/virajmavani/semi-auto-image-annotation-tool
- 영역 분할(segmentation) - U-Net : [unet_segementation.ipynb](material/deep_learning/unet_segementation.ipynb)
- 얼굴 인식(face recognition) : [Face_Recognition.ipynb](material/deep_learning/Face_Recognition.ipynb)
- GAN 이해하기 : [deep_learning_intro.pptx](material/deep_learning//deep_learning_intro.pptx), [dcgan_mnist.ipynb](material/deep_learning/dcgan_mnist.ipynb)
- RNN 이해하기
- 강화학습 이해하기


<br>

## 5일차

- Kaggle 문제 풀이 리뷰
- [딥러닝을 사용한 논문 리뷰](https://docs.google.com/presentation/d/1SZ-m4XVepS94jzXDL8VFMN2dh9s6jaN5fVsNhQ1qwEU/edit?usp=sharing)
- 딥러닝 프로젝트 계획 리뷰
- 딥러닝 채용 공고 리뷰


<br>

## 기타

교육환경, numpy, pandas, matplot
- 교육 환경 : [env.md](material/env.md)
- numpy : 데이터 로딩, 보기, 데이터 변환, 형태 변경 : [library.md](material/library.md)
- linux 기본 명령어 : 
    - bash, cd, ls, rm, mkdir, mv, tar, unzip
    - docker, pip, apt, wget, EVN, git
    - 교육 자료
        - [linux.md](material/linux.md)
        - [linux_exercise.md](material/linux_exercise.md)
- pandas, matplot : [ibrary.md](material/library.md)


기타
- [디노이징 AutoEncoder](material/deep_learning/denoising_autoencoder.ipynb)
- [흥미로운 딥러닝 결과](material/deep_learning/some_interesting_deep_learning.pptx)
- [yolo를 사용한 실시간 불량품 탐지 사례](material/deep_learning/yolo_in_field.mp4)
- [GAN을 사용한 생산설비 이상 탐지](material/deep_learning/anomaly_detection_using_gan.pptx)
- [이상탐지 동영상](material/deep_learning/drillai_anomaly_detect.mp4)


