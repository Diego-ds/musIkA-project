import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dir = "dataset\clean_lyrics_lemmatized.csv"

df = pd.read_csv(dir)
print(len(df))
#pop_threshold = 64
#rock_threshold = 53
# initialize counters
pop_threshold = 54 #thresholds for 50%
rock_threshold = 45

for song_index in df.index:
    song = df.iloc[song_index]
    popularity = song["popularity"]
    genre = song["genre"]
    if genre=="Pop":
        if popularity>=pop_threshold:
            df.at[song_index,'hit'] = 1
        else:
            df.at[song_index,'hit'] = 0
    else:
        if popularity>=rock_threshold:
            df.at[song_index,'hit'] = 1
        else:
            df.at[song_index,'hit'] = 0

df['hit'] = df['hit'].astype(np.int64)
print(df.head())
print(df.info())


df.to_csv("clean_lyrics_lemmatizedHits50P.csv")