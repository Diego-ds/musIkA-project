import requests
from bs4 import BeautifulSoup
import csv
from selenium import webdriver 
from webdriver_manager.chrome import ChromeDriverManager
from time import sleep 
from selenium.webdriver.chrome.options import Options

from unidecode import  unidecode
from urllib.parse import unquote

import pandas as pd
import time
import random

headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:76.0) Gecko/20100101 Firefox/76.0'}

dup = pd.read_pickle("duppk")
print(dup.head())
songs_dic = {}
for i in range(len(dup)):
  r = dup.iloc[i]
  songs_dic[r['id']] = "https://www.google.com/search?client=opera-gx&q={}+{}+lirycs&sourceid=opera&ie=UTF-8&oe=UTF-8".format(r['artist'], r['name'])

lyrics = {}

print(songs_dic["36Owb6DDJbBFXi86x3X61z"])

a=0

for id in songs_dic.keys():
    
    if a==2:
        break
    start = time.time()
    time.sleep(random.uniform(3.0,7.0))
    soup = BeautifulSoup( requests.get(songs_dic[id], headers=headers).content, 'html.parser' )
    items = soup.find_all('span',{'jsname':'YS01Ge'})

    lyrics[id]=""
    print(a)
    testid = id
    a+=1
    for each in items:
        try:
            lyrics[id]+=unidecode(unquote(each.text+"\n"))
        except:
            pass
    end = time.time()
    print(end-start)

print(lyrics[testid])

#    if len(items)==0:
#        print(songs_dic[id])
#        break
print(lyrics)
dataframe = pd.DataFrame.from_dict(lyrics, orient ='index',columns=['lyric'])

dataframe.to_pickle("./lyrics2.pkl")
dataframe.to_csv("./lyrics2.csv")