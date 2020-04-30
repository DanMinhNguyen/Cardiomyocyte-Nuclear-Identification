from tensorflow.keras import backend as K
from tensorflow.keras.constraints import Constraint

class WeightStandardization(Constraint):
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std
    def __call__(self, w):
        return self.mean + ((w - K.mean(w)) * self.std / (K.std(w) + 1e-7))

# class KernelNormalization(Constraint):
#     def __call__(self, w):
#         return K.stack([w[...,ii]/(K.sqrt(K.sum(K.square(w[...,ii])))+1e-7) for ii in range(K.int_shape(w)[-1])],axis=-1)
#         # wlist = []
#         # for ii in range(K.int_shape(w)[-1]):
#         #     wlist.append(w[...,ii] / (K.sqrt(K.sum(K.square(w[...,ii]))) + 1e-7))
#         # wstack = K.stack(wlist,axis=-1)
#         # return wstack


#How maxnorm is defined, for educational purposes
# class MaxNormEducation(Constraint):
#     def __call__(self, w):
#         norms = K.sqrt(K.sum(K.square(w), axis=self.axis, keepdims=True))
#         desired = K.clip(norms, 0, self.max_value)
#         w *= (desired / (K.epsilon() + norms))
#         return w