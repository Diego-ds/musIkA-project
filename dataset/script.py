import pandas as pd

df_features = pd.read_csv("dataset\EsEnSongsDatasetWithFeatures.csv")
df_final = pd.read_csv("dataset\FinalDatasetEsEnSongs.csv")

print(len(df_features))
print(len(df_final))

indexes_to_delete = []

for song_index in df_features.index:
    song = df_features.iloc[song_index]
    id = song["id"]

    condition = id not in df_final["id"].values
    if condition:
        indexes_to_delete.append(song_index)

for index in indexes_to_delete:
    df_features.drop(index,axis=0,inplace=True)
print(len(indexes_to_delete))
print(len(df_features))

df_features.to_csv("./FinalEsEnSongsDatasetWithHighLevelFeatures.csv")