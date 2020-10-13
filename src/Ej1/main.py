import csv
import sys
from kohonen import Kohonen
from oja import Oja
from file_reader import get_json_data

def is_number(elem):
    try:
        float(elem)
        return True
    except ValueError:
        return False

def parse_csv(path):
    data = open(path, newline='')
    reader = csv.reader(data, delimiter=',', quotechar='|', quoting=csv.QUOTE_NONE)
    parsed_data = [[] for x in range(28)]
    parsed_country_names = ['' for x in range(28)]
    idx = 0
    for row in reader:
        if idx > 0:
            for elem in row:
                if is_number(elem):
                    parsed_data[idx - 1].append(float(elem))
                else:
                    parsed_country_names[idx - 1] = elem.replace('"', '')

        idx += 1
    return parsed_country_names, parsed_data

params = get_json_data()
if params[0].lower() == 'kohonen':
    learning_rate = params[1]
    neighborhood_size = params[2]
    path = params[3]
    names, data = parse_csv(path)
    k = Kohonen(names, data, learning_rate, neighborhood_size)
    print('Training network...')
    k.train()
    print('Training finished')
    print('Testing...')
    k.test()
if params[0].lower() == 'oja':
    learning_rate = params[1]
    path = params[3]
    names, data = parse_csv(path)
    k = Oja(names, data, learning_rate)
    print('Training network...')
    k.train()
    print('Training finished')
    print('Testing...')
    k.test(names, data)
