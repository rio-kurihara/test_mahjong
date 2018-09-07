import numpy as np
import keras.backend as K
from keras.engine.topology import Layer
from keras.engine.topology import InputSpec


class L2Normalization(Layer):
    """
    """
    def __init__(self, scale, **kwargs):
        self.scale = scale
        self.gamma = None
        self.axis = None
        if K.image_dim_ordering() == "tf":
            self.axis = 3
        else:
            self.axis = 1
        super(L2Normalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (input_shape[self.axis], )
        self.gamma = K.variable(self.scale * np.ones(shape),
                                name="{}_gamma".format(self.name))
        self.trainable_weights = [self.gamma]

    def call(self, x, mask=None):
        output = K.l2_normalize(x, self.axis)
        output *= self.gamma
        return output
