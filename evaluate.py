import os
import pickle
import numpy as np
from data.get_data import get_rectangles_im
from Population import Population
from datetime import datetime
import tensorflow as tf
from tensorflow.python.layers.layers import conv2d, max_pooling2d, dense, flatten, batch_normalization, dropout
from Layers import ConLayer, PoolLayer, FullLayer
from Individual import Individual


class Evaluate:
    def __init__(self, pops, input_shape, epochs, batch_size, train_data_length, validate_data_length, is_valid=True,
                 index='0'):
        self._pops = pops
        self._input_shape = input_shape
        self._epochs = epochs
        self._batch_size = batch_size
        self._train_data_length = train_data_length
        self._validate_data_length = validate_data_length
        self._epochs_offset = 0
        self._is_valid = is_valid
        self.index = index

    def parse_populations(self, epochs_offset):
        history_best_score = 0
        self._epochs_offset = max(0, epochs_offset)
        self._epochs_offset = int(epochs_offset/5)
        self._epochs += self._epochs_offset
        for population_index in range(self._pops.get_populations_size()):
            individual = self._pops.get_populations_at(population_index)
            try:
                mean_, std_, num_connections_, new_best = \
                    self._parse_individual(individual, population_index, history_best_score)
            except tf.errors.ResourceExhaustedError:
                mean_ = 0.0
                std_ = 0.0
                num_connections_ = 1000000
                new_best = history_best_score
                print('error')
            individual.set_performance_score(mean_, std_, num_connections_)
            with open(os.path.join(os.path.abspath(r'./history-' + self.index + r'/individual_history'),
                                   '{}-{}.dat'.format(epochs_offset, population_index)), 'wb') as f:
                pickle.dump(individual, f)
            history_best_score = new_best

    def _parse_individual(self, individual, ind_index, history_best_score):
        tf.reset_default_graph()
        (train_init_op, train_next_element), (test_init_op, test_next_element) = \
            get_rectangles_im(self._epochs, self._batch_size, is_valid=self._is_valid)
        try:
            is_training, train_op, accuracy, cross_entropy, num_connections = self._build_graph(
                individual, ind_index, train_next_element, test_next_element)
        except IOError:
            individual.training_history['train_acc'] = [0.0]
            individual.training_history['train_loss'] = [0.0]
            individual.training_history['test_acc'] = [0.0]
            individual.training_history['test_loss'] = [0.0]
            mean_test_accuracy = 0.0
            test_accuracy_list = [0.0]
            num_connections = 1e10
            return mean_test_accuracy, np.std(test_accuracy_list), num_connections, history_best_score
        print('the current epochs is {}'.format(self._epochs))
        gpu_options = tf.GPUOptions(allow_growth=True)
        try:
            with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
                sess.run(tf.global_variables_initializer())
                sess.run([train_init_op, test_init_op])
                steps_in_each_epoch = self._train_data_length // self._batch_size
                train_total_steps = int(self._epochs*steps_in_each_epoch)
                test_total_steps = self._validate_data_length // self._batch_size
                train_acc = []
                train_loss = []
                test_acc = []
                test_loss = []
                for step in range(train_total_steps):
                    _, accuracy_str, loss_str = sess.run([train_op, accuracy, cross_entropy], {is_training: True})
                    if step % (2*steps_in_each_epoch) == 0 or step == train_total_steps - 1:
                        test_loss_list = []
                        test_accuracy_list = []
                        for _ in range(test_total_steps):
                            test_accuracy_str, test_loss_str = sess.run([accuracy, cross_entropy], {is_training: False})
                            test_accuracy_list.append(test_accuracy_str)
                            test_loss_list.append(test_loss_str)
                        mean_test_accuracy = np.mean(test_accuracy_list)
                        mean_test_loss = np.mean(test_loss_list)
                        test_acc.append(mean_test_accuracy)
                        test_loss.append(mean_test_loss)
                        train_acc.append(accuracy_str)
                        train_loss.append(loss_str)
                        print('{}, {: 3d}, ind:{: 3d}, step:{: 5d}/{: 5d}, train_loss:{:.6f}, acc:{:.5f}, test_loss:{:.6f}, acc:{:.5f}'.
                              format(datetime.now(), step//steps_in_each_epoch, ind_index, step, train_total_steps,
                                     loss_str, accuracy_str, mean_test_loss, mean_test_accuracy))
                individual.training_history['train_acc'] = train_acc
                individual.training_history['train_loss'] = train_loss
                individual.training_history['test_acc'] = test_acc
                individual.training_history['test_loss'] = test_loss
                if mean_test_accuracy > history_best_score:
                    history_best_score = mean_test_accuracy
        except:
            individual.training_history['train_acc'] = [0.0]
            individual.training_history['train_loss'] = [0.0]
            individual.training_history['test_acc'] = [0.0]
            individual.training_history['test_loss'] = [0.0]
            mean_test_accuracy = 0.0
            test_accuracy_list = [0.0]
            num_connections = 1e10
        return mean_test_accuracy, np.std(test_accuracy_list), num_connections, history_best_score

    def _build_graph(self, individual, ind_index, train_next_element, test_next_element):
        is_training = tf.placeholder(tf.bool, [])
        train_image, train_label = train_next_element
        test_image, test_label = test_next_element
        x = tf.cond(is_training, lambda: train_image, lambda: test_image, name='data_images')
        y_ = tf.cond(is_training, lambda: train_label, lambda: test_label)
        true_y = tf.cast(y_, tf.int64, name='data_labels')
        name_base = 'I_{}'.format(ind_index)
        num_connections = 0
        last_output_feature_map_size = self._input_shape[-1]
        output_list = [x]
        block_name = 1
        current_type = ['', 0]
        for layer_index in range(individual.get_layer_size()):
            current_layer = individual.get_layer_at(layer_index)
            with tf.variable_scope(name_base + 'block_{}'.format(block_name)):
                if current_layer.type == 1:
                    if current_type[0] == 'conv':
                        current_type[1] += 1
                    else:
                        current_type = ['conv', 1]
                    initializer_fn = current_layer.get_initializer()
                    conv_ = conv2d(output_list[-1], current_layer.feature_size, current_layer.filter_size,
                                   padding='same', activation=None, name='conv_{}'.format(current_type[1]),
                                   kernel_initializer=initializer_fn())
                    output_list.append(conv_)
                    # batch norm and active function
                    if current_layer.order > 0.5:
                        # batch norm
                        if current_layer.batch_norm > 0.5:
                            bn_ = batch_normalization(output_list[-1], training=is_training,
                                                      name='conv_{}_bn'.format(current_type[1]))
                            output_list.append(bn_)
                        active_function = current_layer.get_active_fn()
                        # active function
                        output_ = active_function(output_list[-1], name='conv_{}_fn'.format(current_type[1]))
                        output_list.append(output_)
                    else:
                        # active function
                        active_function = current_layer.get_active_fn()
                        output_ = active_function(output_list[-1], name='conv_{}_fn'.format(current_type[1]))
                        output_list.append(output_)
                        # batch norm
                        if current_layer.batch_norm > 0.5:
                            bn_ = batch_normalization(output_list[-1], training=is_training,
                                                      name='conv_{}_bn'.format(current_type[1]))
                            output_list.append(bn_)
                    # dropout
                    if current_layer.dropout_rate >= 0.1:
                        output_ = dropout(output_list[-1], rate=current_layer.get_dropout_rate(), training=is_training,
                                          name='conv_{}_dropout'.format(current_type[1]))
                        output_list.append(output_)
                    width = current_layer.filter_size[0]
                    height = current_layer.filter_size[1]
                    feature_size = current_layer.feature_size
                    num_connections += last_output_feature_map_size*width*height*feature_size + feature_size
                    last_output_feature_map_size = feature_size
                elif current_layer.type == 2:   # Pool
                    block_name += 1
                    current_type = ['pool', 1]
                    pool_ = max_pooling2d(output_list[-1], current_layer.kernel_size, (2, 2), name='pool')
                    output_list.append(pool_)
                    # dropout
                    if current_layer.dropout_rate >= 0.1:
                        pool_ = dropout(output_list[-1], rate=current_layer.get_dropout_rate(), training=is_training,
                                        name='pool_dropout')
                        output_list.append(pool_)
                    num_connections += 0
                    last_output_feature_map_size = last_output_feature_map_size
                elif current_layer.type == 3:   # Full
                    if current_type[0] == 'full':
                        current_type[1] += 1
                        input_data = output_list[-1]
                        input_dim = current_layer.input_size
                    else:
                        current_type = ['full', 1]
                        tmp = output_list[-1]
                        input_data = flatten(output_list[-1])
                        input_dim = current_layer.input_size[0]*current_layer.input_size[1]*current_layer.input_size[2]
                    initializer_fn = current_layer.get_initializer()
                    full_ = dense(input_data, current_layer.hidden_num, activation=None,
                                  name='full_{}'.format(current_type[1]),
                                  kernel_initializer=initializer_fn())
                    output_list.append(full_)
                    if not layer_index == individual.get_layer_size() - 1:
                        # batch norm and active function
                        if current_layer.order > 0.5:
                            # batch norm
                            if current_layer.batch_norm > 0.5:
                                bn_ = batch_normalization(output_list[-1], training=is_training,
                                                          name='full_{}_bn'.format(current_type[1]))
                                output_list.append(bn_)
                            # active function
                            active_function = current_layer.get_active_fn()
                            output_ = active_function(output_list[-1], name='full_{}_fn'.format(current_type[1]))
                            output_list.append(output_)
                        else:
                            # active function
                            active_function = current_layer.get_active_fn()
                            output_ = active_function(output_list[-1], name='full_{}_fn'.format(current_type[1]))
                            output_list.append(output_)
                            # batch norm
                            if current_layer.batch_norm > 0.5:
                                bn_ = batch_normalization(output_list[-1], training=is_training,
                                                          name='full_{}_bn'.format(current_type[1]))
                                output_list.append(bn_)
                        # dropout
                        if current_layer.dropout_rate >= 0.1:
                            output_ = dropout(output_list[-1], rate=current_layer.get_dropout_rate(), training=is_training,
                                              name='full_{}_dropout'.format(current_type[1]))
                            output_list.append(output_)
                    num_connections += input_dim*current_layer.hidden_num + current_layer.hidden_num
                else:
                    raise NameError('No unit with type value: {}'.format(current_layer.type))
        name_scope = '{}_loss'.format(name_base)
        with tf.name_scope(name_scope):
            logits = output_list[-1]
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                cross_entropy = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_y, logits=logits))
        name_scope = '{}_accuracy'.format(name_base)
        with tf.name_scope(name_scope):
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), true_y), tf.float32))
        name_scope = '{}_train'.format(name_base)
        with tf.name_scope(name_scope):
            if self._is_valid:
                train_op = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(cross_entropy)
            else:
                train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cross_entropy)
        return is_training, train_op, accuracy, cross_entropy, num_connections
