import pandas as pd
import numpy as np
from tqdm import tqdm
import librosa
import csv
import pandas as pd



# labels = pd.read_csv('csv/labels.csv', header = None)
# mfcc = pd.read_csv('csv/mfcc.csv', header = None)
# chromagram = pd.read_csv('csv/chromagram.csv', header = None)
# tonnetz = pd.read_csv('csv/tonnetz.csv', header = None)
# mel = pd.read_csv('csv/mel.csv', header = None)

feature = pd.read_csv('csv/feature.csv')
