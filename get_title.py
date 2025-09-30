#gets the video title and channel name for each video in the database
import requests
import sqlite3
import re
from tqdm import tqdm

API_KEY= ""

def clean(text):
    preserve=re.compile(r'[^a-zA-Z0-9\s,.!?\[\]\'\"():/-]')
    text=preserve.sub("", text)
    return text

#fetch the video title and channel username
def fetch_data(video_id):
    url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet&id={video_id}&key={API_KEY}"
    response = requests.get(url)
    title=""
    channel_name=""

    if response.status_code == 200:
        data = response.json()
        if "items" in data and len(data["items"]) > 0:
            video_data = data["items"][0]["snippet"]
            title = clean(video_data["title"])
            channel_name = clean(video_data["channelTitle"])
    return title, channel_name


conn=sqlite3.connect("video_data.db")
cursor=conn.cursor()
cursor.execute("SELECT video_id FROM transcripts")
videos=cursor.fetchall()

for id in tqdm(videos):
    data=fetch_data(id[0])
    cursor.execute("UPDATE transcripts SET title=?, channel=? WHERE video_id=?", (data[0],data[1],id[0]))
    conn.commit()
conn.close()
