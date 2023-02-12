from mutagen.mp3 import MP3
from pydub import AudioSegment
AudioSegment.converter = "ffmpeg.exe"
import os

def split(filename):
    audio = MP3(filename)
    length = audio.info.length
    if length/60 > 2.1:
        sound = AudioSegment.from_mp3(filename)
        start = length/2 * 1000
        window = 60 * 1000
        extract = sound[start-window:start+window]
        extract.export("portion.mp3", format="mp3")

    
split("trippie.mp3")
split("uzi.mp3")


