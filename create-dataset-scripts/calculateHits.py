import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dir = "dataset\FinalDatasetEsEnSongs.csv"

df = pd.read_csv(dir)
print(df.index)
print(df.head())
popularity_range = range(0, 101)  # 0 to 3.5 minutes
# initialize counters
total_counts = [0] * len(popularity_range)
exceed_counts = [0] * len(popularity_range)

for song_index in df.index:

    song = df.iloc[song_index]
    popularity = song["popularity"]
    #if song["genre"] == "Pop": # delete comentary to take into account only certain music gender
    #    continue

    for i in range(len(popularity_range)):
        total_counts[i] += 1
        if popularity > popularity_range[i]:
                exceed_counts[i] += 1

# calculate the percentage of files that exceed the duration threshold for each second of the duration range
percentages = [0] * len(popularity_range)
for i in range(len(popularity_range)):
    percentages[i] = exceed_counts[i] / total_counts[i] * 100

# find the first index where the percentage goes under 95%
index_under_10 = next((i for i, x in enumerate(percentages) if x < 10), None)

# create a line chart
fig, ax = plt.subplots()
ax.plot(popularity_range, percentages, color='blue')
ax.plot([0, 100], [10, 10], color='red', linestyle='dashed')
ax.fill_between(popularity_range, percentages, 10, where=(np.array(percentages) >= 10), interpolate=True, color='green', alpha=0.3)
ax.set_title("Percent of Rock songs exceeding popularity")
ax.set_xlabel("Popularity")
ax.set_ylabel("Percentage of songs")
ax.set_xlim([0, 100])
ax.set_ylim([0, 100])

# add a vertical line to mark where the percentage goes under 95%
if index_under_10 is not None:
    ax.axvline(x=popularity_range[index_under_10], color='black', linestyle='dashed')
    print(popularity_range[index_under_10])

plt.show()