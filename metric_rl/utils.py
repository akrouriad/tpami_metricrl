import os
import pickle


def save_parameters(log_name, params):
    file_name = os.path.join(log_name, 'parameters.pkl')
    with open(file_name, 'wb') as f:
        pickle.dump(params, f)

