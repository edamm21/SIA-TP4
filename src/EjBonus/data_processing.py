import pandas as pd
import matplotlib.pyplot as plt
import statistics
import numpy as np

def pre_process(grid):
    normalized_columns = []
    for i in range(len(grid[0])):    
        entire_col = [row[i] for row in grid]
        media = sum(entire_col) / len(entire_col)
        std = statistics.stdev(entire_col)
        normalized_column = [(value - media) / std for value in entire_col]
        normalized_columns.append(normalized_column)
    array = np.array([np.array(xi) for xi in normalized_columns])
    array = array.transpose()
    return array


CSV_PATH = 'files/europe.csv'

my_csv = CSV_PATH
df = pd.read_csv(my_csv)
features = ['Area','GDP','Inflation','Life.expect','Military','Pop.growth','Unemployment']
values = df[features].to_numpy()
standarized = pre_process(values)
standarized = pd.DataFrame(data=standarized)
standarized.plot(kind='box')
plt.show()

