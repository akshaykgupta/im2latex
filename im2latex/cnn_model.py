from theano import tensor
import numpy
from blocks.bricks import (MLP, Rectifier, Initializable, Sequence,
                           Softmax, Activation, BatchNormalization)
from blocks.bricks.conv import (Convolutional, ConvolutionalSequence,
                                Flattener, MaxPooling)
from blocks.initialization import Constant, Uniform

class CNNEncoder(object):

    def __init__(self, batch_norm, num_channels = 1, **kwargs):

        self.layers = []
        self.layers.append(Convolutional(filter_size = (3, 3), num_filters = 64, border_mode = (1, 1), name = 'conv_1'))
        self.layers.append(Rectifier())
        self.layers.append(MaxPooling(pooling_size = (2, 2), name = 'pool_1'))
        self.layers.append(Convolutional(filter_size = (3, 3), num_filters = 128, border_mode = (1, 1), name = 'conv_2'))
        self.layers.append(Rectifier())
        self.layers.append(MaxPooling(pooling_size = (2, 2), name = 'pool_2'))
        self.layers.append(Convolutional(filter_size = (3, 3), num_filters = 256, border_mode = (1, 1), name = 'conv_3'))
        if batch_norm:
            self.layers.append(BatchNormalization(broadcastable = (False, True, True), name = 'bn_1'))        
        self.layers.append(Rectifier())
        self.layers.append(Convolutional(filter_size = (3, 3), num_filters = 256, border_mode = (1, 1), name = 'conv_4'))
        self.layers.append(Rectifier())
        self.layers.append(MaxPooling(pooling_size = (1, 2), step = (1, 2), name = 'pool_3'))
        self.layers.append(Convolutional(filter_size = (3, 3), num_filters = 512, border_mode = (1, 1), name = 'conv_5'))
        if batch_norm:
            self.layers.append(BatchNormalization(broadcastable = (False, True, True), name = 'bn_2'))
        self.layers.append(Rectifier())
        self.layers.append(MaxPooling(pooling_size = (2, 1), step = (2, 1), name = 'pool_4'))
        self.layers.append(Convolutional(filter_size = (3, 3), num_filters = 512, border_mode = (1, 1), name = 'conv_6'))
        if batch_norm:
            self.layers.append(BatchNormalization(broadcastable = (False, True, True), name = 'bn_3'))
        self.layers.append(Rectifier())
        self.conv_sequence = ConvolutionalSequence(self.layers, 1)
