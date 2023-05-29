import random
import time
import pandas as pd
from lyricsgenius import Genius

genius = Genius('d_tZ1PPmRiml2Vm7MknfLMuFFEM5ouQz3j7JhhBkmUAkuaReqzv-IMHs-0s43OER')
genius.verbose = False # Turn off status messages
genius.remove_section_headers = True # Remove section headers (e.g. [Chorus]) from lyrics when searching
genius.skip_non_songs = True # Include hits thought to be non-songs (e.g. track lists)
genius.excluded_terms = ["(Remix)", "(Live)"] # Exclude songs with these words in their title

dup = pd.read_csv("searchLyric.csv")
print(len(dup))

split = dup.iloc[2000:3000,:]
songs_dic = {}
for i in range(len(split)):
  r = split.iloc[i]
  songs_dic[r['id']] = [r['name'], r['artist']]

lyrics = {}
a=0

start = time.time()
for id in songs_dic.keys():
    try:
      time.sleep(random.uniform(4.0,8.0))
      songs = genius.search_songs(songs_dic[id][0] +' '+songs_dic[id][1])
      #url = song['result']['url']
      #song_lyrics = genius.lyrics(song_url=url)
    
      idg = songs['hits'][0]['result']['id']
      song_lyrics = genius.lyrics(idg)
      print(song_lyrics)

      lyrics[id] = song_lyrics


      print('song number:',a)
      a+=1
      print(id)
      #print(lyrics[id])

    except:
      pass
end = time.time()
print(end-start)


#    if len(items)==0:
#        print(songs_dic[id])
#        break
print(len(lyrics.values()))
dataframe = pd.DataFrame.from_dict(lyrics, orient ='index',columns=['lyrics'])

dataframe.to_pickle("./geniusLyrics3.pkl")
dataframe.to_csv("./geniusLyrics3.csv")