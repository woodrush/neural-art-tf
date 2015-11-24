import tensorflow as tf
import numpy as np

class Network(object):
    def __init__(self, input, params_path):                
        self.params = np.load(params_path).item()
        self.vars = []
        self.vardict = {}
        self.batch_size = int(input.get_shape()[0])
        self.add_('input', input)
        self.setup()

    def setup(self):
        raise NotImplementedError('Must be subclassed.')

    def get_unique_name_(self, prefix):        
        id = sum(t.startswith(prefix) for t,_ in self.vars)+1
        return '%s_%d'%(prefix, id)

    def add_(self, name, var):
        self.vars.append((name, var))
        self.vardict[name] = var

    def get_output(self):
        return self.vars[-1][1]

    def conv(self, h, w, c_i, c_o, stride=1, name=None):
        name = name or self.get_unique_name_('conv')
        with tf.variable_scope(name) as scope:
            weights = self.params[name][0].astype(np.float32)
            conv = tf.nn.conv2d(self.get_output(), weights, [stride]*4, padding='SAME')
            if len(self.params[name]) > 1:
                biases = self.params[name][1].astype(np.float32)
                bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
                relu = tf.nn.relu(bias, name=scope.name)
            else:
                relu = tf.nn.relu(conv, name=scope.name)            
            self.add_(name, relu)
        return self

    def pool(self, size=2, stride=2, name=None):
        name = name or self.get_unique_name_('pool')
        # pool = tf.nn.avg_pool(self.get_output(),
        pool = tf.nn.max_pool(self.get_output(),
                              ksize=[1, size, size, 1],
                              strides=[1, stride, stride, 1],
                              padding='SAME',
                              name=name)
        self.add_(name, pool)
        return self
