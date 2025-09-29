API_KEY=""

#summarising text
from transformers import pipeline, AutoTokenizer
def clean(text):
    text=text.replace("[UNK]", "")
    text=text.replace("’", "'")
    text=text.replace("â€™","'")
    return text

summarizer=pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=0)
Stokenizer=AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6",add_prefix_space=True)

#takes transcript as input and outputs summarised transcript which has roughly target_length characters
def summarise(text):
    tokens=Stokenizer.encode(text)
    max_tokens_per_chunk=400
    chunks=[tokens[i:i+max_tokens_per_chunk] for i in range(0,len(tokens), max_tokens_per_chunk)]
    text_chunks=[Stokenizer.decode(chunk,skip_special_tokens=True) for chunk in chunks]
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

#punctuating
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification
import numpy as np
checkpoint = "unikei/distilbert-base-re-punctuate"
Ptokenizer = DistilBertTokenizerFast.from_pretrained(checkpoint)
Pmodel = DistilBertForTokenClassification.from_pretrained(checkpoint)
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

#getting the summarised and punctuated transcript 
from youtube_transcript_api import YouTubeTranscriptApi
def getText(dictionary):
    return dictionary['text']

def getTranscript(video_id):
    transcript=" ".join([getText(*a) for a in tuple(zip(YouTubeTranscriptApi.get_transcript(video_id)))])
    transcript=punctuate(transcript, Ptokenizer, Pmodel)
    if len(transcript)<500:
        return ""
    if len(transcript)<1500:
        return clean(transcript)
    if len(transcript)>13000:
        return summarise(clean(transcript[:13000]))
    return summarise(clean(transcript))

def getTranscript2(video_id): #for the updated version of Youtube-transcript-api
    ytt=YouTubeTranscriptApi()
    fetched= ytt.fetch(video_id)
    transcript=""
    for snippet in fetched:
        transcript+=snippet.text+" "
    transcript=transcript.replace("\n"," ")
    transcript=punctuate(transcript, Ptokenizer, Pmodel)
    if len(transcript)<500:
        return ""
    if len(transcript)<1500:
        return clean(transcript)
    if len(transcript)>13000:
        return summarise(clean(transcript[:13000]))
    return summarise(clean(transcript))

#model
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

model = T5ForConditionalGeneration.from_pretrained('./fine-tuned-t5-v8')
tokenizer = T5Tokenizer.from_pretrained('./fine-tuned-t5-v8')
def contextualise(title,channel_name,transcript):
    return f"Title: {title} | Channel: {channel_name} | Transcript: {transcript}"
def generate_comment(contextual_input):
    inputs = tokenizer(contextual_input, return_tensors='pt', max_length=512, truncation=True)
    #inputs = tokenizer(contextual_input, return_tensors='pt', max_length=512, truncation=True).to(device)
    outputs = model.generate(inputs['input_ids'], max_length=128, num_beams=5,repetition_penalty=3.0, early_stopping=True, do_sample=True,temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


#fetching title, channel name
import requests
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




video_id="lGJEihgN4OU"
transcript=getTranscript2(video_id)
data=fetch_data(video_id)
contextual_input=contextualise(data[0],data[1],transcript)
print(contextual_input)
text=generate_comment(contextual_input)
print(text)
