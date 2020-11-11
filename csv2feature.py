import pandas as pd
import numpy as np
from tqdm import tqdm
import librosa
import csv
import pandas as pd

# labels = []
# with open('csv/labels.csv', 'r') as csvfile:
#     rows = csv.reader(csvfile)
#     for row in rows:
#         if row[0]!= 'classID':
#             labels.append(row[0])

df = pd.read_csv('csv/labels.csv', header = None)
df = df[1:]
df.to_csv('new.csv', header=0, index=0)
