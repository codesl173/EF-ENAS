import os
import pickle


class IndividualHistory:
    def __init__(self, path=None, index='0'):
        self.individual_dict = {}
        if not path:
            if not os.path.exists(os.path.join('./history-'+index)):
                os.mkdir(os.path.join('./history-'+index))
            path = os.path.join(os.path.abspath('./'), 'history-'+index, 'individual_history')
            if not os.path.exists(path):
                os.mkdir(path)
            self.path = os.path.join(path, 'individual_dict.dat')
        else:
            self.path = os.path.abspath(path)

    def load_individual_history(self):
        with open(self.path, 'rb') as f:
            self.individual_dict = pickle.load(f)

    def save_individual_history(self, ):
        with open(self.path, 'wb') as f:
            pickle.dump(self.individual_dict, f)

    def check_id(self, individual_id):
        if individual_id in self.individual_dict:
            if len(self.individual_dict[individual_id]) < 10:
                return 'ok'
            else:
                return 'no'
        else:
            return False

    def get_id(self, individual_id):
        message = self.individual_dict[individual_id]
        return message

    def add_id(self, individual_id, value):
        if individual_id in self.individual_dict:
            self.individual_dict[individual_id].append(value)
        else:
            self.individual_dict[individual_id] = [value]

    def extend_id_list(self, individual_dict):
        for key in individual_dict:
            if key in self.individual_dict:
                self.individual_dict[key].extend(individual_dict[key])
            else:
                self.individual_dict[key] = individual_dict[key]
