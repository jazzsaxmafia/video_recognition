#-*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import yaml
import ipdb
import cPickle

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
from make_optical_dataset import *
from train_util import sgd
from cnn_util import crop_image

srng = RandomStreams()


class Optic_CNN():

    def __init__(self, n_channel=2, batch_size=50, lstm_dim=500, output_dim=101):

        self.n_channel=n_channel
        self.batch_size = batch_size
        self.lstm_dim = lstm_dim
        self.output_dim = output_dim

        self.conv1_W = initializations.uniform((96, n_channel, 11, 11))
        self.conv1_b = shared_zeros((96,))

        self.conv2_W = initializations.uniform((256, 96, 5, 5))
        self.conv2_b = shared_zeros((256,))

        self.conv3_W = initializations.uniform((384, 256, 3, 3))
        self.conv3_b = shared_zeros((384,))

        self.conv4_W = initializations.uniform((384, 384, 3, 3))
        self.conv4_b = shared_zeros((384,))

        self.conv5_W = initializations.uniform((256, 384, 3, 3))
        self.conv5_b = shared_zeros((256,))

        self.fc6_W = initializations.uniform((9216, 4096))
        self.fc6_b = shared_zeros((4096,))

        self.fc7_W = initializations.uniform((4096, 4096))
        self.fc7_b = shared_zeros((4096,))

        self.lstm_W = initializations.uniform((4096, lstm_dim*4))
        self.lstm_U = initializations.uniform((lstm_dim, lstm_dim*4))
        self.lstm_b = shared_zeros((lstm_dim*4,))

        self.output_W = initializations.uniform((lstm_dim, output_dim))
        self.output_b = shared_zeros((output_dim,))

        self.params = [
                self.conv1_W, self.conv1_b,
                self.conv2_W, self.conv2_b,
                self.conv3_W, self.conv3_b,
                self.conv4_W, self.conv4_b,
                self.conv5_W, self.conv5_b,
                self.fc6_W, self.fc6_b,
                self.fc7_W, self.fc7_b,
                self.lstm_W, self.lstm_U, self.lstm_b,
                self.output_W, self.output_b
                ]

    def get_convpool(self, img, kerns, conv_b, subsample, border_mode, pooling, ws=None, stride=None, normalizing=False,group=1):

        if group == 1:
            conv_out = dnn.dnn_conv(
                    img=img,
                    kerns=kerns,
                    subsample=subsample,
                    border_mode=border_mode
                    )

            conv_out += conv_b.dimshuffle('x',0,'x','x')
        else:
            filter_shape = kerns.shape

            n_output = filter_shape[0] / 2
            n_input = filter_shape[1] / 2

            conv_out1 = dnn.dnn_conv(
                    img=img[:,:n_input,:,:],
                    kerns=kerns[:n_output, :n_input, :,:],
                    subsample=subsample,
                    border_mode=border_mode)

            conv_out2 = dnn.dnn_conv(
                    img=img[:,n_input:,:,:],
                    kerns=kerns[n_output:, n_input:,:,:],
                    subsample=subsample,
                    border_mode=border_mode)

            conv_out = T.concatenate([conv_out1, conv_out2], axis=1)
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

    def get_dropout(self, X, p=0.5):
        if p > 0:
            retain_rate = 1 - p
            X *= srng.binomial(X.shape, p=retain_rate, dtype=theano.config.floatX)
            X /= retain_rate

        return X

    def get_fc(self, input, fc_W, fc_b):
        fc_out = T.dot(input, fc_W) + fc_b
        fc_out = T.maximum(fc_out, 0)

        #drop_out = self.get_dropout(fc_out, p=0.5)
        return fc_out#drop_out

    def get_visual(self, images):

        convpool1_out = self.get_convpool(
                img=images,
                kerns=self.conv1_W,
                conv_b=self.conv1_b,
                subsample=(4,4),
                border_mode=0,
                pooling=True,
                ws=(3,3),
                stride=(2,2),
                normalizing=True,
                group=1
                )

        convpool2_out = self.get_convpool(
                img=convpool1_out,
                kerns=self.conv2_W,
                conv_b=self.conv2_b,
                subsample=(1,1),
                border_mode=2,
                pooling=True,
                ws=(3,3),
                stride=(2,2),
                normalizing=True,
                group=2
                )

        convpool3_out = self.get_convpool(
                img=convpool2_out,
                kerns=self.conv3_W,
                conv_b=self.conv3_b,
                subsample=(1,1),
                border_mode=1,
                pooling=False,
                normalizing=False,
                group=1
                )

        convpool4_out = self.get_convpool(
                img=convpool3_out,
                kerns=self.conv4_W,
                conv_b=self.conv4_b,
                subsample=(1,1),
                border_mode=1,
                pooling=False,
                normalizing=False,
                group=2
                )

        convpool5_out = self.get_convpool(
                img=convpool4_out,
                kerns=self.conv5_W,
                conv_b=self.conv5_b,
                subsample=(1,1),
                border_mode=1,
                pooling=True,
                ws=(3,3),
                stride=(2,2),
                normalizing=True,
                group=2
                )

        fc6_out = self.get_fc(T.flatten(convpool5_out, 2), self.fc6_W, self.fc6_b)
        fc7_out = self.get_fc(fc6_out, self.fc7_W, self.fc7_b)


        return (
                fc7_out
                )

    def forward_lstm(self, x, mask):
        def _slice(_x, n, dim):
            return _x[:, n*dim:(n+1)*dim]

        def _step(m_tm_1, x_t, h_tm_1, c_tm_1):
            lstm_preactive = T.dot(h_tm_1, self.lstm_U) + T.dot(x_t, self.lstm_W) + self.lstm_b

            i = T.nnet.sigmoid(_slice(lstm_preactive, 0, self.lstm_dim))
            f = T.nnet.sigmoid(_slice(lstm_preactive, 1, self.lstm_dim))
            o = T.nnet.sigmoid(_slice(lstm_preactive, 2, self.lstm_dim))
            c = T.tanh(_slice(lstm_preactive, 3, self.lstm_dim))

            c = f*c_tm_1 + i*c
            c = m_tm_1[:,None]*c + (1.-m_tm_1)[:,None]*c_tm_1

            h = o*T.tanh(c)
            h = m_tm_1[:,None]*h + (1.-m_tm_1)[:,None]*h_tm_1

            return [h,c]

        h0 = T.alloc(0., x.shape[1], self.lstm_dim)
        c0 = T.alloc(0., x.shape[1], self.lstm_dim)

        rval, updates = theano.scan(
                fn=_step,
                sequences=[mask,x],
                outputs_info=[h0,c0]
                )

        h_list, c_list = rval
        return h_list

    def build_model(self, lr=0.005):
        tensor5 = T.TensorType('float32', (False,)*5) # ( batch, time, channel, width, height )
        image_sequences = tensor5('image_sequences')
        labels = T.imatrix('labels')
        masks = T.matrix('masks') # (batch, time)

        image_sequences_shuffle = image_sequences.dimshuffle(1,0,2,3,4)
        masks_shuffle = masks.dimshuffle(1,0)

        fc7s, updates = theano.scan(
                fn=lambda imgs: self.get_visual(imgs),
                sequences=[image_sequences_shuffle]
                ) # ( batch, time, 4096 )

        h_list = self.forward_lstm(fc7s, masks_shuffle)
        h_last = h_list[-1]

        output = T.dot(h_last, self.output_W) + self.output_b
        probs = T.nnet.softmax(output)

        cost = T.nnet.categorical_crossentropy(probs, labels)
        cost = T.mean(cost)

        updates = sgd(cost=cost, params=self.params, lr=lr)

        f_train = theano.function (
                inputs=[image_sequences, masks, labels],
                outputs=cost,
                updates=updates,
                allow_input_downcast=True)
#        f_train = theano.function_dump (
#                'dumpfile.bin',
#                inputs=[image_sequences, masks, labels],
#                outputs=cost,
#                updates=updates,
#                allow_input_downcast=True)

        return f_train

def main():
    video_path = '../../data/UCF_video'
    all_videos = os.listdir(video_path)
    all_video_path = map(lambda x: os.path.join(video_path, x), all_videos)
    all_labels = map(lambda x: x.split('.')[0].split('_')[1], all_videos)

    unique_labels = np.unique(all_labels)
    label_dict = pd.Series(range(len(unique_labels)), index=unique_labels)

    interval = 10
    n_epochs = 300
    batch_size = 20
    dim_output = 101
    lr = 0.005

    optic_cnn = Optic_CNN()
#    with open('../../models/epoch_4.pickle') as f:
#        optic_cnn = cPickle.load(f)

    f_train = optic_cnn.build_model(lr=lr)

    for epoch in range(n_epochs):
        for start, end in zip \
            (
                range(0, len(all_videos) + batch_size, batch_size),
                range(batch_size, len(all_videos)+batch_size, batch_size)
            ):

            batch_vids = all_video_path[start:end]
            batch_labels = label_dict.ix[np.array(all_labels[start:end])].values

            batch_frames = map(lambda vid: read_video(vid, interval), batch_vids)
            batch_lens = map(lambda frms: len(frms), batch_frames)
            current_batch_size = len(batch_vids)
            maxlen = np.max(batch_lens)

            frame_tensor = np.zeros((current_batch_size, maxlen, 2, 227, 227))
            mask_matrix = np.zeros((current_batch_size, maxlen))
            label_matrix = np.zeros((current_batch_size, dim_output))

            for idx, frame in enumerate(batch_frames):
                frame_tensor[idx][:len(frame)] = frame
                mask_matrix[idx][:len(frame)] = 1
                label_matrix[idx][batch_labels[idx]] = 1

            cost = f_train(frame_tensor, mask_matrix, label_matrix)
            print cost

        with open('../../models.v2/epoch_'+str(epoch)+'.pickle', 'w') as f:
            cPickle.dump(optic_cnn, f)

        if np.mod(epoch, 5) == 0 and epoch != 0:
            lr *= 0.98
            f_train = optic_cnn.build_model(lr=lr)
