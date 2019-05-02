import librosa
import os
import numpy as np

train_dir = "data/train_data"
test_dir = "data/test_files"
to_normelaize = True
transalte = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5}
axis_to_normelaize = 1

two_axis_normaliztion = 0


def normalize(v, by_axis = axis_to_normelaize):
    global ll
    sum_col = np.mean(v, axis=by_axis).reshape(v.shape[1 - by_axis], -1)
    std_col = np.std(v, axis=by_axis).reshape(v.shape[1 - by_axis], -1)
    if by_axis == 1:
        res = (v - sum_col) / std_col
    else:
        res = (v - sum_col.T) / std_col.T

    if two_axis_normaliztion == 1 and by_axis == 1:
        return normalize(res, 1 - by_axis)
    return res


def eudis(mfcc, sound):
    dist = (mfcc - sound) ** 2
    dist = np.sum(dist, axis=1)
    dist = np.sqrt(dist)
    return dist


def DTW(mfcc, sound):
    def find_shortest_path(matrix):
        path = np.zeros_like(matrix)
        top_i = len(matrix)
        top_j = len(matrix)
        path[0] = np.cumsum(matrix[0])
        path[:, 0] = np.cumsum(matrix[:, 0]).T
        for j in range(1, top_j):
            for i in range(1, top_i):
                path[i, j] = matrix.item(i, j)
                path[i, j] += np.min([path.item(i - 1, j - 1), path.item(i - 1, j), path.item(i, j - 1)])
                # which_min = np.argmin( [path.item(i-1,j-1), path.item(i-1,j),path.item(i,j-1)])
                # loc = [(i-1,j-1),(i-1,j),(i,j-1)][which_min]
                # path[i, j] += path.item(loc)
        return path.item(top_i - 1, top_j - 1)

    assert mfcc.shape == sound.shape
    mat = []
    for v in mfcc:
        dist = eudis(sound, v)
        mat.append(dist)
    matrix = np.asmatrix(mat)
    return find_shortest_path(matrix)


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
                if to_normelaize:
                    mfcc = normalize(mfcc)
                digit_list = train.get(digit_name, [])
                digit_list.append(mfcc)
                train[digit_name] = digit_list
    for digit, item in train.items():
        assert len(item) == 5
    return train


from collections import Counter

euclid = Counter()
dtw = Counter()
text_to_file = []
gold_text = []
train_data = prepare_train_data(train_dir)
for test_file in os.listdir(test_dir):
    if not test_file.endswith(".wav"): continue
    euclid_scores = {}
    dtw_scores = {}
    row_text = test_file
    test_filename = os.path.join(test_dir, test_file)
    y, sr = librosa.load(test_filename, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    for digit, items in train_data.items():
        eculid_distance = []
        DTW_distance = []
        for wav_train_sound in items:
            if to_normelaize:
                mfcc = normalize(mfcc)

            eculid_distance.append(eudis(mfcc.T, wav_train_sound.T).mean())
            DTW_distance.append(DTW(mfcc.T, wav_train_sound.T))
        eculid_distance = min(eculid_distance)
        DTW_distance = min(DTW_distance)

        euclid_scores[digit] = eculid_distance
        dtw_scores[digit] = DTW_distance

    sorted_euclid = sorted(euclid_scores.items(), key=lambda euclid_scores: euclid_scores[1])
    sorted_dtw = sorted(dtw_scores.items(), key=lambda dtw_scores: dtw_scores[1])

    euclid.update([transalte[sorted_euclid[0][0]]])
    dtw.update([transalte[sorted_dtw[0][0]]])
    text_to_file.append(
        row_text + ' - ' + str(transalte[sorted_euclid[0][0]]) + ' - ' + str(transalte[sorted_dtw[0][0]]) + "\n")

open("output.txt", 'w').writelines(text_to_file)

print(euclid)
print(dtw)
