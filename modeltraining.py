import pickle
import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture
from featureextraction import extract_features
# from speakerfeatures import extract_features
import warnings

warnings.filterwarnings("ignore")

# path to training data
# source   = "development_set/"
source = "trainingData/"

# path where training speakers will be saved

# dest = "speaker_models/"
# train_file = "development_set_enroll.txt"

dest = "Speakers_models/"
#train_file = "trainingDataPath.txt"
train_file = "ourTraining.txt"
file_paths = open(train_file, 'r')

count = 1
# Extracting features for each speaker (5 files per speakers)
features = np.asarray(())
for path in file_paths:
    #readfile
    path = path.strip()
    print(path)

    # read the audio
    sr, audio = read(source + path)

    # extract 40 dimensional MFCC & delta MFCC features#featureextraction.py需要去理解MFCC之運作
    vector = extract_features(audio, sr)

    if features.size == 0:
        features = vector
    else:
        features = np.vstack((features, vector))
    # when features of 5 files of speaker are concatenated, then do model training
    # -> if count == 5: --> edited below
    #print(count)
    if count == 20:

        gmm = GaussianMixture(n_components=16, max_iter=200, covariance_type='diag', n_init=3)
        gmm.fit(features)

        # dumping the trained gaussian model#picklefile用於python特有的型別和python的資料型別間進行轉換
        picklefile = path.split("-")[0] + (".gmm")
        pickle.dump(gmm, open(dest + picklefile, 'wb'))
        print(('+ modeling completed for speaker:', picklefile, " with data point = ", features.shape))
        features = np.asarray(())
        count = 0
    count = count + 1
