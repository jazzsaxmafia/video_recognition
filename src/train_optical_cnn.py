#-*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import yaml

import cv2
from theano_alexnet.alex_net import *

with open('config.yaml') as f:
    config = yaml.load(f)

batch_size = config['batch_size']
model = AlexNet(config)

(
    train_function,
    train_model, # all dummy
    validate_model,
    train_error,
    learning_rate,
    shared_x,
    shared_y,
    rand_arr,
    vels
) = compile_models(model, config)

#train_function(image, labels, learning_rate, np.random.randn(3))
# image : (n_channel, height, width, batch_size)
#         (    2    ,  227  ,  227 ,    100    )

video_patha = '../data/UCF_video'
interval = 10
n_epochs = 10

for epoch in range(n_epochs):
    for start, end in zip
        (
            range(0, len(train_set) + batch_size, batch_size),
            range(batch_size, len(train_set)+batch_size, batch_size)
        ):

        batch_data = train_set[start:end]



