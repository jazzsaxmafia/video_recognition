# -*- coding: utf-8 -*-
import theano
import theano.tensor as T

import ipdb
import os
from cnn_util import *

from keras import initializations
from keras.utils.theano_utils import shared_zeros

import pandas as pd
import numpy as np

def sgd( cost, params, lr ):
    grads = T.grad(cost, params)
    updates = []
    for param, grad in zip(params, grads):
        updates.append((param, param - lr*grad))

    return updates

class LSTM():
    def __init__(self, dim_input, dim, dim_output):
        self.dim_input = dim_input
        self.dim = dim
        self.dim_output = dim_output

        self.lstm_U = initializations.orthogonal((self.dim, self.dim*4))
        self.lstm_W = initializations.uniform((self.dim_input, self.dim*4))
        self.lstm_b = shared_zeros((self.dim*4))

        self.output_W = initializations.uniform((self.dim, self.dim_output))
        self.output_b = shared_zeros((self.dim_output))

        self.params = [self.lstm_W, self.lstm_U, self.lstm_b,
                       self.output_W, self.output_b]

    def forward_lstm(self, x, mask):
        def _slice(_x, n, dim):
            return _x[:, n*dim:(n+1)*dim]

        def _step(m_tm_1, x_t, h_tm_1, c_tm_1):
            lstm_preactive = T.dot(h_tm_1, self.lstm_U) + T.dot(x_t, self.lstm_W) + self.lstm_b

            i = T.nnet.sigmoid(_slice(lstm_preactive, 0, self.dim))
            f = T.nnet.sigmoid(_slice(lstm_preactive, 1, self.dim))
            o = T.nnet.sigmoid(_slice(lstm_preactive, 2, self.dim))
            c = T.tanh(_slice(lstm_preactive, 3, self.dim))

            c = f*c_tm_1 + i*c
            c = m_tm_1[:,None] * c + (1.-m_tm_1)[:,None] * c_tm_1

            h = o * T.tanh(c)
            h = m_tm_1[:,None] * h + (1.-m_tm_1)[:,None] * h_tm_1

            return [h,c]

        h0 = T.alloc(0., x.shape[1], self.dim) # (n_samples, dim)
        c0 = T.alloc(0., x.shape[1], self.dim)

        rval, updates = theano.scan(
                fn=_step,
                sequences=[mask,x],
                outputs_info=[h0,c0]
                )

        h_list, c_list = rval
        return h_list

    def build_model(self):
        x = T.tensor3('x') # (n_samples, n_frames, dim_image)
        mask = T.matrix('mask')
        label = T.imatrix('label')

        x_shuffled = x.dimshuffle(1,0,2) #(n_frames, n_samples, dim_image)
        mask_shuffled = mask.dimshuffle(1,0)

        h_list = self.forward_lstm(x_shuffled, mask_shuffled)
        h_last = h_list[-1]

        output = T.dot(h_last, self.output_W) + self.output_b
        probs = T.nnet.softmax(output)
        cost = T.nnet.categorical_crossentropy(probs, label)
        cost = T.mean(cost)

        updates = sgd(cost=cost, params=self.params, lr=0.001)

        f_train = theano.function(
                inputs=[x,mask,label],
                outputs=cost,
                updates=updates,
                allow_input_downcast=True)

        return f_train

def read_video(vid, interval):
    cap = cv2.VideoCapture(vid)

    frame_count = 0
    frame_list = []
    while True:
        ret, frame = cap.read()
        if ret is False:
            break

        if np.mod(frame_count , interval) == 0:
            frame_list.append(frame)

        frame_count += 1

    frame_array = np.array(frame_list)
    return frame_array

def make_dataset(video_path):
    videos = os.listdir(video_path)

    videos = map(lambda x: os.path.join(video_path, x), videos)
    labels = np.unique(map(lambda x: x.split('/')[-1].split('_')[1], videos))
    label_dict = pd.Series(range(len(labels)), index=labels)

    all_labels = map(lambda x: x.split('/')[-1].split('_')[1], videos)
    all_label_class = label_dict[np.array(all_labels)]

    dataset = pd.DataFrame({'video':videos, 'label':all_label_class.values})
    train_size = int(dataset.shape[0]*0.8)

    train = dataset[:train_size]
    test = dataset[train_size:]

    train.to_pickle('../data/train_set.pickle')
    test.to_pickle('../data/test_set.pickle')
    label_dict.to_pickle('../data/label_dict.pickle')

    ipdb.set_trace()

    return train, test, label_dict

def main():

    video_path = '../data/UCF_video'

    if not os.path.exists('../data/train_set.pickle'):
        train_set, test_set, label_dict = make_dataset(video_path)
    else:
        train_set = pd.read_pickle('../data/train_set.pickle')
        test_set = pd.read_pickle('../data/test_set.pickle')
        label_dict = pd.read_pickle('../data/label_dict.pickle')

    interval = 10
    batch_size = 10
    dim_input=4096
    dim=256
    n_epochs = 100
    dim_output=len(label_dict)

    cnn = CNN()
    lstm = LSTM(dim_input=dim_input, dim=dim, dim_output=dim_output)

    f_train = lstm.build_model()

    for epoch in range(n_epochs):
        for start, end in zip(
                range(0, len(train_set)+batch_size, batch_size),
                range(batch_size, len(train_set)+batch_size, batch_size)
            ):

            batch_data = train_set[start:end]

            batch_files = batch_data['video'].values
            batch_labels = batch_data['label'].values

            batch_frames = map(lambda vid: read_video(vid, interval), batch_files)
            batch_features = map(lambda vid: cnn.get_features(vid), batch_frames)
            batch_lens = map(lambda feat: feat.shape[0], batch_features)
            maxlen = np.max(batch_lens)

            frame_tensor = np.zeros((len(batch_files), maxlen, 4096))
            mask_matrix = np.zeros((len(batch_files), maxlen))
            label_matrix = np.zeros((len(batch_files), dim_output))

            for idx,frame in enumerate(batch_features):
                frame_tensor[idx][:len(frame)] = frame
                mask_matrix[idx][:len(frame)] = 1
                label_matrix[idx][batch_labels[idx]] = 1

            cost = f_train(frame_tensor, mask_matrix, label_matrix)
            print cost
