import librosa
import numpy as np
import pandas as pd
from time import process_time

def load_song(id):
    y, sr = librosa.load('data/songs/{}.mp3'.format(id))
    return calculate_features(y, sr, id)

def calculate_features(y, sr, id):

    times = {"id": id}

    chroma_t_s = process_time()
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_t_e = process_time()

    times["chroma_stft"] = (chroma_t_e-chroma_t_s)

    mel_t_s = process_time()
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_t_e = process_time()

    times["mel"] = (mel_t_e-mel_t_s)

    mfcc_t_s = process_time()
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    mfcc_t_e = process_time()

    times["mfcc"] = (mfcc_t_e-mfcc_t_s)

    rms_t_s = process_time()
    rms = librosa.feature.rms(y=y)
    rms_t_e = process_time()

    times["rms"] = (rms_t_e-rms_t_s)

    spec_centroids_t_s = process_time()
    spec_centroids = librosa.feature.spectral_centroid(y=y, sr =sr)
    spec_centroids_t_e = process_time()

    times["spec_centroids"] = (spec_centroids_t_e-spec_centroids_t_s)

    spec_bw_t_s = process_time()
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr =sr)
    spec_bw_t_e = process_time()

    times["spec_bw"] = (spec_bw_t_e-spec_bw_t_s)

    spec_contrast_t_s = process_time()
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr =sr)
    spec_contrast_t_e = process_time()

    times["spec_contrast"] = (spec_contrast_t_e-spec_contrast_t_s)

    spec_flatness_t_s = process_time()
    spec_flatness = librosa.feature.spectral_flatness(y=y)
    spec_flatness_t_e = process_time()

    times["spec_flatness"] = (spec_flatness_t_e-spec_flatness_t_s)

    spec_rolloff_t_s = process_time()
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr =sr)
    spec_rolloff_t_e = process_time()

    times["spec_rolloff"] = (spec_rolloff_t_e-spec_rolloff_t_s)

    tonnetz_t_s = process_time()
    tonnetz = librosa.feature.tonnetz(y=y, sr =sr)
    tonnetz_t_e = process_time()

    times["tonnetz"] = (tonnetz_t_e-tonnetz_t_s)

    zero_crossing_rate_t_s = process_time()
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
    zero_crossing_rate_t_e = process_time()

    times["zero_crossing_rate"] = (zero_crossing_rate_t_e-zero_crossing_rate_t_s)

    chromas = {}
    for i in enumerate(chroma_stft):
        chromas[f'chroma_{i[0]}'] = np.mean(chroma_stft[i[0]])

    mel_coef = {}
    for i in enumerate(mel):
        mel_coef[f'mel_{i[0]}'] = np.mean(mel[i[0]])

    mel_ceps = {}
    for i in enumerate(mfcc):
        mel_ceps[f'mfcc_{i[0]}'] = np.mean(mfcc[i[0]])

    tonn = {}
    for i in enumerate(tonnetz):
        tonn[f'tonnetz_{i[0]}'] = np.mean(tonnetz[i[0]])
    
    contrast = {}
    for i in enumerate(spec_contrast):
        contrast[f'spec_contrast_{i[0]}'] = np.mean(spec_contrast[i[0]])

    base = {}
    base['id'] = id
    base['rms'] = np.mean(rms[0])
    base['spec_centroids'] = np.mean(spec_centroids[0])
    base['spec_bw'] = np.mean(spec_bw[0])
    base['spec_flatness'] = np.mean(spec_flatness[0])
    base['spec_rolloff'] = np.mean(spec_rolloff[0])
    base['zero_crossing_rate'] = np.mean(zero_crossing_rate[0])
    
    row = { **base, **contrast, **chromas, **mel_coef, **mel_ceps, **tonn }

    return row, times

data = pd.read_csv('lowLevelFeatures.csv')
song_dataset = pd.read_csv("EnEsSongsDataset.csv")

times_data = pd.read_csv("times_low_level_features.csv")

for i in range(len(song_dataset)):
  row = song_dataset.iloc[i]
  llf_row, times = load_song(row['id'])
  times_data = times_data.append(times, ignore_index=True)
  data = data.append(llf_row, ignore_index=True)

data.to_csv('lowLevelFeatures.csv')
times_data.to_csv('times_low_level_features.csv')


