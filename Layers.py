import math
from tensorflow.python.ops.nn import relu, crelu, tanh, elu
from tensorflow.python.ops.init_ops import glorot_normal_initializer, he_normal

Xavier = glorot_normal_initializer


class ConLayer:
    def __init__(self, input_size, feature_size, batch_norm, active_function, order, initializer, dropout_rate):
        self.input_size = input_size
        self.output_size = [input_size[0], input_size[1], feature_size]
        self.filter_size = [3, 3]
        self.feature_size = feature_size
        self.batch_norm = batch_norm
        self.active_function = active_function
        self.order = order
        self.initializer = initializer
        self.dropout_rate = dropout_rate
        self.score = 0.0
        self.type = 1

    def get_bn(self):
        return True if self.batch_norm > 0.5 else False

    def get_active_fn_name(self):
        if self.active_function < 0.25:
            return 'crelu'
        elif self.active_function < 0.5:
            return 'relu'
        elif self.active_function < 0.75:
            return 'elu'
        else:
            return 'tanh'

    def get_active_fn(self):
        if self.active_function < 0.25:
            return crelu
        elif self.active_function < 0.5:
            return relu
        elif self.active_function < 0.75:
            return elu
        else:
            return tanh

    def get_initializer(self):
        return Xavier if self.initializer < 0.5 else he_normal

    def get_initializer_name(self):
        return 'Xavier' if self.initializer < 0.5 else 'he_normal'

    def get_dropout_rate(self):
        return math.floor(self.dropout_rate*100)/100

    def __str__(self):
        if self.order > 0.5:
            message = 'C[num:{}, BN:{}, fn:{}, init:{}, dp:{}]'.format(
                self.feature_size, self.get_bn(), self.get_active_fn_name(), self.get_initializer_name(),
                self.get_dropout_rate())
        else:
            message = 'C[num:{}, fn:{}, BN:{}, init:{}, dp:{}]'.format(
                self.feature_size, self.get_active_fn_name(), self.get_bn(), self.get_initializer_name(),
                self.get_dropout_rate())
        return message


class PoolLayer:
    def __init__(self, input_size, dropout_rate):
        self.input_size = input_size
        self.output_size = [int(input_size[0]/2), int(input_size[1]/2), input_size[2]]
        self.kernel_size = [2, 2]
        self.dropout_rate = dropout_rate
        self.score = 0.0
        self.type = 2

    def get_dropout_rate(self):
        return math.floor(self.dropout_rate*100)/100

    def __str__(self):
        message = 'P[dp:{}]'.format(self.get_dropout_rate())
        return message


class FullLayer:
    def __init__(self, input_size, hidden_num, batch_norm, active_function, order, initializer, dropout_rate):
        self.input_size = input_size
        self.output_size = hidden_num
        self.hidden_num = hidden_num
        self.batch_norm = batch_norm
        self.active_function = active_function
        self.order = order
        self.initializer = initializer
        self.dropout_rate = dropout_rate
        self.score = 0.0
        self.type = 3

    def get_bn(self):
        return True if self.batch_norm > 0.5 else False

    def get_active_fn_name(self):
        if self.active_function < 0.25:
            return 'crelu'
        elif self.active_function < 0.5:
            return 'relu'
        elif self.active_function < 0.75:
            return 'elu'
        else:
            return 'tanh'

    def get_active_fn(self):
        if self.active_function < 0.25:
            return crelu
        elif self.active_function < 0.5:
            return relu
        elif self.active_function < 0.75:
            return elu
        else:
            return tanh

    def get_initializer(self):
        return Xavier if self.initializer < 0.5 else he_normal

    def get_initializer_name(self):
        return 'Xavier' if self.initializer < 0.5 else 'he_normal'

    def get_dropout_rate(self):
        return math.floor(self.dropout_rate*100)/100

    def __str__(self):
        if self.order > 0.5:
            message = 'F[num:{}, BN:{}, fn:{}, init:{}, dp:{}]'.format(
                self.hidden_num, self.get_bn(), self.get_active_fn_name(), self.get_initializer_name(),
                self.get_dropout_rate()
            )
        else:
            message = 'F[num:{}, fn:{}, BN:{}, init:{}, dp:{}]'.format(
                self.hidden_num, self.get_active_fn_name(), self.get_bn(), self.get_initializer_name(),
                self.get_dropout_rate()
            )
        return message
