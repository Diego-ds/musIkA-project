# -*- coding: utf-8 -*-
"""SongsDownloadYoutube.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1uEX54x2gJpETtSpyrSTZufWpcChBqcse
"""

!pip install pytube
!pip install youtube-search

from youtube_search import YoutubeSearch
from pytube import YouTube
import os
import shutil
import pandas as pd
import datetime

dir = "data/songs/"

def search(search, artist):
  video_search = search+"--"+artist+" official audio"
  print(video_search)
  results = YoutubeSearch(video_search, max_results=30).to_dict()
  for video in results:
    artist = artist.split(",")[0]
    #print(artist+"--"+video["channel"])
    try:
      #checks that the video is less than 10 minutes to avoid recopilations or bucles
      video_duration = datetime.datetime.strptime(video["duration"],'%M:%S')
      max_duration = datetime.datetime.strptime("10:00",'%M:%S')
      duration_comparison = video_duration.time() < max_duration.time()
      #checks that se song is actually the one that is being searched
      same_song = search.lower() in video["title"].lower()
      
      #or artist in video["title"]
      if((artist in video["channel"]) and duration_comparison and same_song):
        url = "https://www.youtube.com/watch?v={}".format(video["id"])
        print(search+"--"+video["title"])
        return url
    except:
      pass
  
  return None

def download(url,download_dir,desired_name):
  # url input from user
  yt = YouTube(url)

  # extract only audio
  video = yt.streams.filter(only_audio=True).first()

  # download the file
  out_file = video.download(output_path=".")

  # save the file
  base, ext = os.path.splitext(out_file)
  new_file = download_dir + desired_name + '.mp3'
  shutil.move(out_file,new_file)
  #os.rename(out_file, new_file)

  # result of success
  print(yt.title + " has been successfully downloaded.")

df = pd.read_csv("/content/EnEsSongsDataset.csv")

counter = 0
for song in range(len(df)):
  row = row = df.iloc[song]
  url = search(row["name"], row["artist"])
  print(url, row["id"])
  if(url is None):
    counter = counter+1
  else:
    download(url,dir,row["id"])

  if(song==10):
    break

print(counter)