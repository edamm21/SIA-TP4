import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir('../../')
import json
with open(os.getcwd() + "/input.json") as file:
    data = json.load(file)
    method = data['METHOD']
    lr = data['LEARNING_RATE']
    n_size = data['NEIGHBORHOOD_SIZE']
    path = data['DATA_PATH']

def get_json_data():
    return [method, lr, n_size, path]