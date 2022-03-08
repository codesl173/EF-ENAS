import numpy as np
from Individual import Individual
from individual_set_dict import IndividualHistory


class Population:
    def __init__(self, input_size, pops_num, x_prob=0.9, x_eta=20, index='0'):
        self.data_shape = input_size
        self._x_prob = x_prob
        self._x_eta = x_eta
        self._current_pops_num = pops_num
        self.populations = []
        self.individual_history = IndividualHistory(index=index)
        for _ in range(pops_num):
            individual = Individual(self.data_shape)
            individual.initialize_random_2()
            self.populations.append(individual)

    def save_history(self, save_to_disk=True):
        self.individual_history.individual_dict = {}
        for individual in self.populations:
            self.individual_history.add_id(individual.get_md5(), individual.mean)
        if save_to_disk:
            self.individual_history.save_individual_history()

    def print_block(self):
        for individual in self.populations:
            layer_message = 'len:{:02d}'.format(len(individual.individual))
            for layer in individual.individual:
                if layer.type == 1:
                    layer_message += ', conv'
                elif layer.type == 2:
                    layer_message += ', pool'
                else:
                    layer_message += ', full'
            print(layer_message)

    def set_populations(self, new_pops):
        self.populations = new_pops
        self._current_pops_num = len(new_pops)

    def extend_populations(self, new_pops):
        self.populations.extend(new_pops)
        self._current_pops_num = len(self.populations)

    def get_populations_size(self):
        return self._current_pops_num

    def get_populations_at(self, population_index):
        return self.populations[population_index]

    def crossover(self, pop1, pop2):
        p1 = pop1
        p2 = pop2
        if self._x_prob:
            p1_length = p1.get_layer_size()
            p2_length = p2.get_layer_size()
            cross_point = np.random.random()
            p1_cross_point = int(np.ceil((p1_length - 1)*cross_point))
            p2_cross_point = int(np.ceil((p2_length - 1)*cross_point))
            if p1.get_layer_at(p1_cross_point).type == 3 and (not p2.get_layer_at(p2_cross_point).type == 3):
                if p1_length > p2_length:
                    while not p2.get_layer_at(p2_cross_point).type == 3:
                        p2_cross_point += 1
                else:
                    while p1.get_layer_at(p1_cross_point).type == 3:
                        p1_cross_point -= 1
            elif (not p1.get_layer_at(p1_cross_point).type == 3) and p2.get_layer_at(p2_cross_point).type == 3:
                if p1_length > p2_length:
                    while p2.get_layer_at(p2_cross_point).type == 3:
                        p2_cross_point -= 1
                else:
                    while not p1.get_layer_at(p1_cross_point).type == 3:
                        p1_cross_point += 1
            p1_layer_list = []
            p2_layer_list = []
            p1_layer_list.extend(p1.individual[:p1_cross_point])
            p2_layer_list.extend(p2.individual[:p2_cross_point])
            p1_layer_list.extend(p2.individual[p2_cross_point:])
            p2_layer_list.extend(p1.individual[p1_cross_point:])
            p1.set_layers(p1_layer_list)
            p2.set_layers(p2_layer_list)
            p1._check_consistency()
            p2._check_consistency()
        return p1, p2

    @staticmethod
    def _get_cell(individual):
        cell_list = []
        cell = [individual.individual[0]]
        for i in range(1, len(individual.individual)):
            if individual.individual[i].type == cell[-1].type:
                cell.append(individual.individual[i])
            else:
                if len(cell) == 1:
                    cell.append(individual.individual[i])
                    cell_list.append(cell)
                    cell = [individual.individual[i]]
                else:
                    cell_list.append(cell)
                    cell = [cell[-1], individual.individual[i]]
                    cell_list.append(cell)
                    cell = [individual.individual[i]]
        cell_list.append(cell)
        return cell_list

    @staticmethod
    def _is_same_cell(cell_1, cell_2):
        if len(cell_1) == len(cell_2):
            for item_1, item_2 in zip(cell_1, cell_2):
                if not item_1.type == item_2.type:
                    return False
            return True
        else:
            return False

    @staticmethod
    def _is_crossover(cell_list_1, cell_list_2):
        cell_1 = []
        cell_2 = []
        if cell_list_1 == [] or cell_list_2 == []:
            return False

        for item in cell_list_1:
            tmp = 0
            if item[0].type == 3:
                tmp = 1
            if item[-1].type == 3:
                tmp += 1
            cell_1.append(tmp)
        for item in cell_list_2:
            tmp = 0
            if item[0].type == 3:
                tmp = 1
            if item[-1].type == 3:
                tmp += 1
            cell_2.append(tmp)
        cell_1_index = list(range(len(cell_1)))
        np.random.shuffle(cell_1_index)
        cell_2_index = list(range(len(cell_2)))
        np.random.shuffle(cell_2_index)
        for item_1 in cell_1_index:
            for item_2 in cell_2_index:
                if cell_1[item_1] == cell_2[item_2]:
                    return cell_list_1[item_1], cell_list_2[item_2]
        return False

    def crossover_base_cell(self, ind1, ind2):
        ind1_cell = self._get_cell(ind1)
        ind2_cell = self._get_cell(ind2)
        ind1_tmp = [ind1_cell[0], ind1_cell[-1]]
        ind2_tmp = [ind2_cell[0], ind2_cell[-1]]
        for i in range(len(ind1_cell)-1, -1, -1):
            for j in range(len(ind2_cell)-1, -1, -1):
                cell_1 = ind1_cell[i]
                cell_2 = ind2_cell[j]
                if self._is_same_cell(cell_1, cell_2):
                    ind1_cell.pop(i)
                    ind2_cell.pop(j)
                    break
        crossover_cells = self._is_crossover(ind1_cell, ind2_cell)
        if not crossover_cells:
            return ind2, ind1
        else:
            crossover_cell_1 = crossover_cells[0]
            crossover_cell_2 = crossover_cells[1]
            if id(crossover_cell_1[0]) == id(ind1.individual[0]):
                crossover_cell_2 = ind2_tmp[0]
            elif id(crossover_cell_2[0]) == id(ind2.individual[0]):
                crossover_cell_1 = ind1_tmp[0]
            # p1
            first = 0
            second = 0
            for i in range(len(ind1.individual)):
                if id(ind1.individual[i]) == id(crossover_cell_1[0]):
                    first = i
                if id(ind1.individual[i]) == id(crossover_cell_1[-1]):
                    second = i+1
                    break
            p1_layer_list = ind1.individual[:first]
            p1_layer_list.extend(crossover_cell_2)
            p1_layer_list.extend(ind1.individual[second:])
            # p2
            for i in range(len(ind2.individual)):
                if id(ind2.individual[i]) == id(crossover_cell_2[0]):
                    first = i
                if id(ind2.individual[i]) == id(crossover_cell_2[-1]):
                    second = i+1
            p2_layer_list = ind2.individual[:first]
            p2_layer_list.extend(crossover_cell_1)
            p2_layer_list.extend(ind2.individual[second:])
        ind1.set_layers(p1_layer_list)
        ind2.set_layers(p2_layer_list)
        ind1._check_consistency()
        ind2._check_consistency()
        return ind1, ind2

    def _sbx(self, v1, v2, xl, xu):
        if np.random.random() < 0.5:
            if abs(v1 - v2) > 1e-14:
                x1 = min(v1, v2)
                x2 = max(v1, v2)
                r = np.random.random()
                beta = 1.0 + (2.0 * (x1 - xl) / (x2 - x1))
                alpha = 2.0 - beta ** -(self._x_eta + 1)
                if r <= 1.0 / alpha:
                    beta_q = (r * alpha) ** (1.0 / (self._x_eta + 1))
                else:
                    beta_q = (1.0 / (2.0 - r * alpha)) ** (1.0 / (self._x_eta + 1))

                c1 = 0.5 * (x1 + x2 - beta_q * (x2 - x1))
                beta = 1.0 + (2.0 * (xu - x2) / (x2 - x1))
                alpha = 2.0 - beta ** -(self._x_eta + 1)
                if r <= 1.0 / alpha:
                    beta_q = (r * alpha) ** (1.0 / (self._x_eta + 1))
                else:
                    beta_q = (1.0 / (2.0 - r * alpha)) ** (1.0 / (self._x_eta + 1))
                c2 = 0.5 * (x1 + x2 + beta_q * (x2 - x1))
                c1 = min(max(c1, xl), xu)
                c2 = min(max(c2, xl), xu)
                if np.random.random() < 0.5:
                    return c2, c1
                else:
                    return c1, c2
            else:
                return v1, v2
        else:
            return v1, v2

    def test_sbx(self):
        xl = 0.0
        xu = 2.0
        x = np.linspace(0.0, 2.1, 100)
        y = np.linspace(0.0, 2.0, 100)
        for x_ in x:
            for y_ in y:
                a, b = self._sbx(x_, y_, xl, xu)
                if (not a == a) or (not b == b):
                    print(a, b)

    def __str__(self):
        message = []
        for index in range(self.get_populations_size()):
            message.append(str(self.get_populations_at(index)))
        return '\n'.join(message)
