import os
import librosa
import librosa.display
import numpy as np
from os import listdir
from os.path import isfile, join
from numpy import save

def generate_ndarray():
    all_num = 0
    #We intialize large numpy matrix
    final_sound_arr = np.empty([2300*50,128,44])
    #We read the files containing the sound data
    onlyfolders_path = [os.path.join(os.getcwd(),f) for f in listdir(os.getcwd()) if not isfile(join(os.getcwd(),f))]
    for name in onlyfolders_path:
        print(name)
        onlyfiles_path = [os.path.join(name,f) for f in listdir(name) if isfile(join(name,f))]
        total=len(onlyfiles_path)
        counter=0
        for path in onlyfiles_path:
            sig, fs = librosa.load(path)
            #Roughly 10% of the data generates wrong dimension results, so we just filter it out:
            if (128,44) == np.shape(librosa.feature.melspectrogram(y=sig, sr=fs, n_fft = 2048)):
                final_sound_arr[all_num]=librosa.feature.melspectrogram(y=sig, sr=fs, n_fft = 2048)
                counter+=1
                all_num+=1
                if counter%100==0:
                    print(counter,"done out of", total)
                #Just in case we set our initial numpy matrix too small we cut the process before we get an error
                elif all_num %(2300*50-1) == 0:
                    break
    #Shuffling is done so that when we train our neural network there will be no large chains of very similar data
    np.random.shuffle(final_sound_arr)
    txt = 'sound_arr' + str(all_num)
    save(txt,final_sound_arr)

            
        
generate_ndarray()
