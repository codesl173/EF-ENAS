import os
import copy

save_path = r'./check_folder'
save_path = os.path.abspath(save_path)
if not os.path.exists(save_path):
    os.mkdir(save_path)
    for i in range(1, 16):
        os.mkdir(os.path.join(save_path, '{:02d}'.format(i)))


def deduplication(pops):
    pops = copy.deepcopy(pops)

    result_list = []
    while pops:
        same_list = []
        individual = pops.pop()
        same_list.append(individual)
        for individual_index in range(len(pops)-1, -1, -1):
            if is_same(individual, pops[individual_index]):
                same_list.append(pops.pop(individual_index))
        mean = 0.0
        std = 0.0
        count = 0
        for individual in same_list:
            mean += (individual.mean*individual.count)
            std += (individual.std*individual.count)
            count += individual.count
        mean = mean/count
        std = std/count
        individual = same_list[0]
        individual.mean = mean
        individual.std = std
        individual.count = count
        result_list.append(individual)

    return result_list


def is_exist_in_pops(network, pops):
    for individual in pops:
        if is_same(network, individual):
            return True
    return False


def is_same(network1, network2):
    network1 = network1.individual
    network2 = network2.individual
    if not len(network1) == len(network2):
        return False

    for layer_index in range(len(network1)-1):
        if not network1[layer_index].type == network2[layer_index].type:
            return False
        if network1[layer_index].type == 1:
            if not _is_same_conv(network1[layer_index], network2[layer_index]):
                return False
        if network1[layer_index].type == 2:
            if not _is_same_pool(network1[layer_index], network2[layer_index]):
                return False
        if network1[layer_index].type == 3:
            if not _is_same_full(network1[layer_index], network2[layer_index]):
                return False
    if not network1[-1].get_initializer() == network2[-1].get_initializer():
        return False
    return True


def is_same_structure(network1, network2):
    network1 = network1.individual
    network2 = network2.individual
    if not len(network1) == len(network2):
        return False
    for layer_index in range(len(network1)):
        if not network1[layer_index].type == network2[layer_index].type:
            return False
    return True


def _is_same_conv(conv1, conv2):
    if not conv1.feature_size == conv2.feature_size:
        return False
    if not conv1.get_bn() == conv2.get_bn():
        return False
    if not conv1.get_active_fn_name() == conv2.get_active_fn_name():
        return False
    if not conv1.order == conv2.order:
        return False
    if not conv1.get_initializer() == conv2.get_initializer():
        return False
    if not conv1.get_dropout_rate() == conv2.get_dropout_rate():
        return False
    return True


def _is_same_pool(pool1, pool2):
    if not pool1.get_dropout_rate() == pool2.get_dropout_rate():
        return False
    return True


def _is_same_full(full1, full2):
    if not full1.hidden_num == full2.hidden_num:
        return False
    if not full1.get_bn() == full2.get_bn():
        return False
    if not full1.get_active_fn_name() == full2.get_active_fn_name():
        return False
    if not full1.order == full2.order:
        return False
    if not full1.get_initializer() == full2.get_initializer():
        return False
    if not full1.get_dropout_rate() == full2.get_dropout_rate():
        return False
    return True


def analyse_structure(populations):
    populations = copy.deepcopy(populations)
    populations_list = []
    while populations:
        same_list = []
        individual = populations.pop()
        same_list.append(individual)
        for individual_index in range(len(populations) - 1, -1, -1):
            if is_same_structure(individual, populations[individual_index]):
                same_list.append(populations.pop(individual_index))
        populations_list.append(same_list)
