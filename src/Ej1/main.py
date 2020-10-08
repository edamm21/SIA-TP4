import csv
import sys
from kohonen import Kohonen

def is_number(elem):
    try:
        float(elem)
        return True
    except ValueError:
        return False

def parseCSV():
    data = open('./data/europe.csv', newline='')
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

if __name__ == '__main__':
    algorithm = sys.argv[1].lower()
    if algorithm == 'kohonen':
        try:
            epochs = int(sys.argv[2])
            lr = float(sys.argv[3])
            names, data = parseCSV()
            k = Kohonen(names, data, lr)
            k.train()
        except IndexError:
            print('Para ejecutar: python(3) main.py Kohonen <tasa_aprendizaje>')

