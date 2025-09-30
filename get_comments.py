#finds best comments under each video
from itertools import islice
from youtube_comment_downloader import *
downloader = YoutubeCommentDownloader()
import sqlite3
import re
from tqdm import tqdm

def clean(text):
    preserve=re.compile(r'[^a-zA-Z0-9\s,.!?\[\]\'\"():/-]')
    text=text.replace("\n", "[BREAK]")
    text=text.replace("😂", "[LAUGH]") #preserves breaks and laughing emoji
    text=preserve.sub("", text)
    text=text.strip()
    return text

def get_comments(video_id):
    comments = downloader.get_comments_from_url('https://www.youtube.com/watch?v='+video_id, sort_by=SORT_BY_POPULAR)
    out=[]
    for comment in islice(comments,100):
        likes=comment['votes']
        if likes[-1]=="K":
            likes=float(likes[:-1])*1000
        else:
            likes=float(likes)
        
        if likes>50: #filters comments with less than 50 likes
            text=clean(comment['text'])
            if not comment['reply']:
                out.append((text,likes))
    return out[1:]

conn=sqlite3.connect("video_data.db")
cursor=conn.cursor()
cursor.execute("SELECT video_id FROM transcripts")
videos=cursor.fetchall()

i=0
for id in tqdm(videos):
    comments=get_comments(id[0])
    for comment in comments:
        i+=1
        cursor.execute("INSERT INTO comments (comment_id, video_id, comment, likes) VALUES (?, ?, ?, ?)", (i, id[0], comment[0], comment[1]))
        conn.commit()


conn.close()
