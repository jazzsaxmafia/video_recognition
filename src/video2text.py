# -*- coding: utf-8 -*-
import theano
import theano.tensor as T

import ipdb
import os
from taeksoo.cnn_util import *

from keras import initializations
from keras.utils.theano_utils import shared_zeros

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
            lstm_preactive = T.dot(h_tm_1, self.lstm_U) + T.dot(x_t, self.lstm_W) + self.b

            i = T.nnet.sigmoid(_slice(lstm_preactive, 0, self.dim))
            f = T.nnet.sigmoid(_slice(lstm_preactive, 1, self.dim))
            o = T.nnet.sigmoid(_slice(lstm_preactive, 2, self.dim))
            c = T.tanh(_slice(lstm_preactive, 3, self.dim))

            c = f*c_tm_1 + i*c
            c = m_tm_1[:,None] * c + (1.-m_tm_1)[:,None] * c_tm_1

            h = o * T.tanh(c)
            h = m_tm_1[:,None] * h + (1.-m_tm_1)[:,None] * h_tm_1

            return [h,c]

        h0 = T.alloc(0., x.shape[0], self.dim)
        c0 = T.alloc(0., x.shape[0], self.dim)

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

        f_train = theano.function(
                inputs=[x,mask,label],
                outputs=cost,
                allow_input_downcast=True)

        return f_train


