import re
from pytube import Playlist
import os


YOUTUBE_STREAM_AUDIO = '140' # modify the value to download a different stream
DOWNLOAD_DIR = os.path.join(os.path.abspath(""), 'yt-vids')

playlist = Playlist('https://www.youtube.com/playlist?list=PLaylJHO985sU1wwjO7Wnfc34IIqQNX8Li')

# this fixes the empty playlist.videos list
playlist._video_regex = re.compile(r"\"url\":\"(/watch\?v=[\w-]*)")

print(f'Amount of videos: {len(playlist.video_urls)}')

for url in playlist.video_urls:
    print(url)

# physically downloading the audio track
for video in playlist.videos:
    audioStream = video.streams.get_by_itag(YOUTUBE_STREAM_AUDIO)
    audioStream.download(output_path=DOWNLOAD_DIR)
