import librosa
import os
import numpy as np


train_dir = "data/train_data"
test_dir = "data/test_files"

transalte = {'one':1, 'two':2, 'three':3 , 'four':4,'five':5}

def eudis(mfcc, sound):
    dist = (mfcc - sound) ** 2
    dist = np.sum(dist, axis=1)
    dist = np.sqrt(dist)
    return dist



def DTW(mfcc, sound):
    def find_shortest_path(matrix):
        i = len(matrix) - 1
        j = len(matrix) - 1
        total = matrix.item(i,j)
        while i > 0 and j >0:
            which_min = np.argmin( [matrix.item(i-1,j-1), matrix.item(i-1,j),matrix.item(i,j-1)])
            if which_min == 0:
                i-=1
                j-=1
                total += matrix.item(i-1,j-1)
            elif which_min == 1:
                i-=1
                total += matrix.item(i-1,j)
            else:
                j-=1
                total += matrix.item(i,j-1)
        while i > 0:
            i -= 1
            total += matrix.item(i - 1, j)
        while j > 0:
            j -= 1
            total += matrix.item(i, j - 1)
        return total


    assert mfcc.shape == sound.shape
    mat = []
    for v in mfcc:
        dist = eudis(v,sound)
        mat.append(dist)
    matrix = np.asmatrix(mat)
    matrix[0,0] = 0
    return find_shortest_path(matrix)

# def get_gold_anntotion(file):
#     import speech_recognition as sr
#     harvard = sr.AudioFile(file)
#     r = sr.Recognizer()
#     with harvard as source:
#         audio = r.record(source)
#     print(file)
#     try:
#         digit = r.recognize_google(audio)
#     except:
#         try:
#             digit = r.recognize_ibm(audio)
#         except:
#             print("isnt known")
#             digit = '0'
#     print(digit)
#     try:
#         return transalte[digit]
#     except:
#         if digit in str(transalte.values()):
#             return digit
#         else:
#             print("error detection file was thought to be ", digit)
#             return "0"
#             exit()

def prepare_train_data(train_directory):
    train = {}
    for subdir, dirs, files in os.walk(train_directory):
        digit_name = subdir.rsplit("/", 1)[-1]
        for file in files:
            # print os.path.join(subdir, file)
            if file.endswith(".wav"):
                filepath = subdir + os.sep + file
                y, sr = librosa.load(filepath, sr=None)
                mfcc = librosa.feature.mfcc(y=y, sr=sr)
                digit_list = train.get(digit_name, [])
                digit_list.append(mfcc)
                train[digit_name] = digit_list
    return train

from collections import Counter
euclid = Counter()
dtw = Counter()
text_to_file = []
gold_text = []
train_data = prepare_train_data(train_dir)
for test_file in os.listdir(test_dir):
    euclid_scores = {}
    dtw_scores = {}
    row_text = test_file
    test_filename = os.path.join(test_dir, test_file)
    if not test_filename.endswith(".wav"): continue
    y, sr = librosa.load(test_filename, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    for digit,items in train_data.items():
        eculid_distance = []
        DTW_distance    = []
        for wav_train_sound in items:
            eculid_distance.append(eudis(mfcc.T, wav_train_sound.T).sum())
            DTW_distance.append(DTW(mfcc.T, wav_train_sound.T))
        eculid_distance = min(eculid_distance)
        DTW_distance = min(DTW_distance)

        euclid_scores[digit] = eculid_distance
        dtw_scores[digit]    = DTW_distance

    sorted_euclid = sorted(euclid_scores.items(), key=lambda euclid_scores: euclid_scores[1])
    sorted_dtw = sorted(dtw_scores.items(), key=lambda dtw_scores: dtw_scores[1])
    # gold_annotation = get_gold_anntotion(test_filename)

    euclid.update([transalte[sorted_euclid[0][0]]])
    dtw.update([transalte[sorted_dtw[0][0]]])
    text_to_file.append(row_text + ' - ' + str(transalte[sorted_euclid[0][0]])  + ' - ' + str(transalte[sorted_dtw[0][0]]) + "\n")
    # gold_text.append(row_text + ' - ' + str(gold_annotation) + "\n")

open("output.txt", 'w').writelines(text_to_file)
# open("gold_output.txt", 'w').writelines(gold_text)
# print(euclid)
# print(dtw)
