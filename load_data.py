# Load various imports 
import pandas as pd
import os
import librosa
import librosa.display

from helpers.WavFileHelper import WavFileHelper
wavfilehelper = WavFileHelper()

metadata = pd.read_csv('metadata/UrbanSound8K.csv')
audiodata = []
i=0
for index, row in metadata.iterrows():
    i+=1   
    if i > 10:
        break
    file_name = os.path.join(os.path.abspath('D:/DATASET/UrbanSound8K/audio/'),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
    data = wavfilehelper.read_file_properties(file_name)
    audiodata.append(data)

# Convert into a Panda dataframe
audiodf = pd.DataFrame(audiodata, columns=['num_channels','sample_rate','bit_depth'])

#print(audiodf.num_channels.value_counts(normalize=True))
# print(audiodf.sample_rate.value_counts(normalize=True))
# print(audiodf.bit_depth.value_counts(normalize=True))
