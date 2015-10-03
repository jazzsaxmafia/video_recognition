#-*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import yaml
import ipdb

import cv2
import os
#from theano_alexnet.alex_net import *
import theano
import theano.tensor as T
from theano.sandbox.cuda import dnn

from keras import initializations
from keras.utils.theano_utils import shared_zeros

from pylearn2.expr.normalize import CrossChannelNormalization
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from theano.tensor.signal.downsample import max_pool_2d
from theano.tensor.nnet import conv2d
import sys
sys.path.append('../util')
from train_util import sgd
from cnn_util import crop_image

srng = RandomStreams()

class CNN():
    def __init__(self, n_channels, batch_size=30):
        self.n_channels = n_channels
        self.batch_size = batch_size

        self.conv1_W = initializations.uniform((96, n_channels, 7,7))
        self.conv1_b = shared_zeros((96,))

        self.conv2_W = initializations.uniform((256,96,5,5))
        self.conv2_b = shared_zeros((256,))

        self.conv3_W = initializations.uniform((512,256,3,3))
        self.conv3_b = shared_zeros((512,))

        self.conv4_W = initializations.uniform((512,512,3,3))
        self.conv4_b = shared_zeros((512,))

        self.conv5_W = initializations.uniform((512,512,3,3))
        self.conv5_b = shared_zeros((512,))



    def get_convpool(self, img, kerns, conv_b, subsample, border_mode, pooling, ws=None, stride=None, normalizing=False):

        conv_out = dnn.dnn_conv(
                img=img,
                kerns=kerns,
                subsample=subsample,
                border_mode=border_mode
                )

        conv_out += conv_b.dimshuffle('x',0,'x','x')
        conv_out = T.maximum(conv_out, 0)

        if pooling:
            pool_out = dnn.dnn_pool(
                    conv_out,
                    ws=ws,
                    stride=stride
                    )
        else:
            pool_out = conv_out

        if normalizing:
            norm_out = CrossChannelNormalization()(pool_out)
        else:
            norm_out = pool_out
        return norm_out

    def get_visual(self, images):
        convpool1_out = self.get_convpool(
                img=images,
                kerns=self.conv1_W,
                conv_b=self.conv1_b,
                subsample=(2,2),
                border_mode=0,
                pooling=True,
                ws=(2,2),
                stride=(2,2),
                normalizing=True
                )
        convpool2_out = self.get_convpool(
                img=convpool1_out,
                kerns=self.conv2_W,
                conv_b=self.conv2_b,
                subsample=(2,2),
                border_mode=0,
                pooling=True,
                ws=(2,2),
                stride=(2,2),
                normalizing=True
                )
        convpool3_out = self.get_convpool(
                img=convpool2_out,
                kerns=self.conv3_W,
                conv_b=self.conv3_b,
                subsample=(1,1),
                border_mode=1,
                pooling=False,
                normalizing=False
                )
        convpool4_out = self.get_convpool(
                img=convpool3_out,
                kerns=self.conv4_W,
                conv_b=self.conv4_b,
                subsample=(1,1),
                border_mode=1,
                pooling=False,
                normalizing=False
                )
        convpool5_out = self.get_convpool(
                img=convpool4_out,
                kerns=self.conv5_W,
                conv_b=self.conv5_b,
                subsample=(1,1),
                border_mode=1,
                pooling=True,
                ws=(2,2),
                stride=(2,2),
                normalizing=True)

        fc6_out = T.dot(T.flatten(convpool5_out, 2), self.fc6_weight) + self.fc6_b
        fc7_out = T.dot(fc6_out, self.fc7_weight) + self.fc7_b

        return (
            convpool1_out,
            convpool2_out,
            convpool3_out,
            convpool4_out,
            convpool5_out,
            fc6_out,
            fc7_out
            )

    def build_model(self):
        images = T.tensor4('images')
        image_feats = self.get_visual(images)

        ff = theano.function(
            inputs=[images],
            outputs=image_feats,
            allow_input_downcast=True)


        return ff
cnn = CNN(n_channels=3)
ff = cnn.build_model()
image_ex = np.random.randn(10,3,224,224)
