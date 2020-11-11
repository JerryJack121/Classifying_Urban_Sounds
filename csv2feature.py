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

labels = pd.read_csv('csv/labels.csv', header = None)
mfcc = pd.read_csv('csv/mfcc.csv', header = None)
chromagram = pd.read_csv('csv/chromagram.csv', header = None)
tonnetz = pd.read_csv('csv/tonnetz.csv', header = None)
mel = pd.read_csv('csv/mel.csv', header = None)

