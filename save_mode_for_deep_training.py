import os
import pickle
from Population import Population
from evaluate import Evaluate
from check_networks import is_same_structure

the_best_individual = []
for index in [1, 2, 3]:
    cache_path = r'./history-{}/cache'.format(index)
    cache_path = os.path.abspath(cache_path)
    populations_list = []
    for i in range(51):
        path = os.path.join(cache_path, 'gen_{:03d}'.format(i), 'pops.dat')
        with open(path, 'rb') as f:
            data = pickle.load(f)['pops'].populations
            populations_list.append(data)
    the_new_population = populations_list[-1][:10]
    the_new_block = []
    max_index = []
    max_dp = 0
    for index_individual, individual in enumerate(the_new_population):
        block = []
        individual_max_dp = 0
        for item in individual.individual:
            if item.type == 1:
                block.append('conv[{:.2f}]'.format(item.get_dropout_rate()))
                if item.get_dropout_rate() > individual_max_dp:
                    individual_max_dp = item.get_dropout_rate()
            elif item.type == 2:
                block.append('pool[{:.2f}]'.format(item.get_dropout_rate()))
            else:
                block.append('full[{:.2f}]'.format(item.get_dropout_rate()))
        block.append(individual.complexity)
        block.append(individual_max_dp)
        block.append(individual.mean)
        the_new_block.append(block)
        if max_dp < individual_max_dp:
            max_dp = individual_max_dp
            max_index = [index_individual]
        elif max_dp == individual_max_dp:
            max_index.append(index_individual)
        print(block)
    if len(max_index) > 1:
        if len(the_new_population[max_index[0]].individual) > len(the_new_population[max_index[1]].individual):
            print('==> ', max_index[0], ' ==> ', the_new_block[max_index[0]])
            the_best_individual.append(the_new_population[max_index[0]])
        elif len(the_new_population[max_index[0]].individual) < len(the_new_population[max_index[1]].individual):
            print('==> ', max_index[1], ' ==> ', the_new_block[max_index[1]])
            the_best_individual.append(the_new_population[max_index[1]])
        else:
            print('==> ', max_index[0], ' ==> ', the_new_block[max_index[0]])
            the_best_individual.append(the_new_population[max_index[0]])
    else:
        print('==> ', max_index[0], ' ==> ', the_new_block[max_index[0]])
        the_best_individual.append(the_new_population[max_index[0]])
    print('-'*16)
if not os.path.exists(r'./the_best_model'):
    os.mkdir(r'./the_best_model')
with open('./the_best_model/the_best_individuals.dat', 'wb') as f:
    pickle.dump(the_best_individual, f)
