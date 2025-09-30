#fetches the transcript of the video.

from youtube_transcript_api import YouTubeTranscriptApi
import torch
import numpy as np
from tqdm import tqdm
import sqlite3

from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification
checkpoint = "unikei/distilbert-base-re-punctuate"
tokenizer = DistilBertTokenizerFast.from_pretrained(checkpoint)
model = DistilBertForTokenClassification.from_pretrained(checkpoint)
encoder_max_length = 256


def split_to_segments(wrds, length, overlap):
    resp = []
    i = 0
    while True:
        wrds_split = wrds[(length * i):((length * (i + 1)) + overlap)]
        if not wrds_split:
            break

        resp_obj = {
            "text": wrds_split,
            "start_idx": length * i,
            "end_idx": (length * (i + 1)) + overlap,
        }

        resp.append(resp_obj)
        i += 1
    return resp

def punctuate_wordpiece(wordpiece, label):
    if label.startswith('UPPER'):
        wordpiece = wordpiece.upper()
    elif label.startswith('Upper'):
        wordpiece = wordpiece[0].upper() + wordpiece[1:]
    if label[-1] != '_' and label[-1] != wordpiece[-1]:
        wordpiece += label[-1]
    return wordpiece

def punctuate_segment(wordpieces, word_ids, labels, start_word):
    result = ''
    for idx in range(0, len(wordpieces)):
        if word_ids[idx] == None:
            continue
        if word_ids[idx] < start_word:
            continue
        wordpiece = punctuate_wordpiece(wordpieces[idx][2:] if wordpieces[idx].startswith('##') else wordpieces[idx],
                            labels[idx])
        if idx > 0 and len(result) > 0 and word_ids[idx] != word_ids[idx - 1] and result[-1] != '-':
            result += ' '
        result += wordpiece
    return result

def process_segment(words, tokenizer, model, start_word):

    tokens = tokenizer(words['text'],
                       padding="max_length",
                       # truncation=True,
                       max_length=encoder_max_length,
                       is_split_into_words=True, return_tensors='pt')
    
    with torch.no_grad():
        logits = model(**tokens).logits
    logits = logits.cpu()
    predictions = np.argmax(logits, axis=-1)

    wordpieces = tokens.tokens()
    word_ids = tokens.word_ids()
    id2label = model.config.id2label
    labels = [[id2label[p.item()] for p in prediction] for prediction in predictions][0]

    return punctuate_segment(wordpieces, word_ids, labels, start_word)

#punctuates text
def punctuate(text, tokenizer, model):
    text = text.lower()
    text = text.replace('\n', ' ')
    words = text.split(' ')
    
    overlap = 50
    slices = split_to_segments(words, 150, 50)
    
    result = ""
    start_word = 0
    for text in slices:
        corrected = process_segment(text, tokenizer, model, start_word)
        result += corrected + ' '
        start_word = overlap
    return result


from transformers import pipeline, AutoTokenizer
import math

def clean(text):
    text=text.replace("[UNK]", "")
    text=text.replace("’", "'")
    text=text.replace("â€™","'")
    return text

#summarisation model
summarizer=pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=0)
tokenizer2=AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6",add_prefix_space=True)

#takes transcript as input and outputs summarised transcript which has roughly target_length characters
def summarise(text,target_length):
    tokens=tokenizer2.encode(text)
    max_tokens_per_chunk=400
    chunks=[tokens[i:i+max_tokens_per_chunk] for i in range(0,len(tokens), max_tokens_per_chunk)]
    text_chunks=[tokenizer2.decode(chunk,skip_special_tokens=True) for chunk in chunks]
    if len(chunks[-1])<200:
        text_chunks[-2]=text_chunks[-2]+text_chunks[-1]
        text_chunks.pop()
    for i in range(1,len(text_chunks)): #makes sure chunks are split at the end of a sentence, not inbetween
        try:
            x=list(text_chunks[i]).index(".")
            if x<400:
                text_chunks[i-1]=text_chunks[i-1]+text_chunks[i][:x+2]
                text_chunks[i]=text_chunks[i][x+2:]
        except:
            pass
    summary=[]
    for chunk in text_chunks:
        summary.append(summarizer(chunk, max_length=100, min_length=30, do_sample=False)[0]['summary_text'])
    return " ".join(summary)


def getText(dictionary):
    return dictionary['text']

#takes video id as input and outputs punctuated transcript
def getTranscript(video_id):
    transcript=" ".join([getText(*a) for a in tuple(zip(YouTubeTranscriptApi.get_transcript(video_id)))])
    transcript=punctuate(transcript, tokenizer, model)
    if len(transcript)<3000:
        return transcript, "NOTGOOD"
    return transcript, summarise(clean(transcript), 3000)

def getTranscript2(video_id): #for the updated version of Youtube-transcript-api
    ytt=YouTubeTranscriptApi()
    fetched= ytt.fetch(video_id)
    out=""
    for snippet in fetched:
        out+=snippet.text+" "
    return out.replace("\n"," ")

con=sqlite3.connect("video_data.db")
cursor=con.cursor()

f=open("videos.txt")
videos=f.readlines()
f.close()

for video in tqdm(videos):
    id=video[:-1]
    try:
        data=getTranscript2(id)
        cursor.execute("INSERT INTO transcripts (video_id, transcript, transcript_summarised) VALUES (?, ?, ?)", (id, data[0], data[1]))
        con.commit()

    except Exception as e:
        print(f"error: {e}")

f.close()


