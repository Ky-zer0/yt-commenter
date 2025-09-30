# Youtube Commenter
The goal of this project was to generate Youtube comments based on a video's transcript, title and the channel name.

The qualities of a 'good' comment are:
- Coherence: Sentences make sense and are grammatically correct, as if written by a human
- Relevance: Uses context from the video, stays on topic
- Complexity: Includes interesting structure instead of simply "I love this video!", for example

Videos were selected from a broad range of topics, and their transcripts were fetched using [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api), and repunctuated using distilbert-base-re-punctuate. To extract only the key information, they were then summarised with distilbart-cnn-12-6. 

For each video, a handful of comments were taken, prioritising comments with more likes. Emojis had to be removed, with the exception of laughing emojis, which were replaced with '[LAUGH]'. Line breaks were replaced with '[BREAK]'.

In the end, after removing videos with limited/no transcript, videos not in English and videos with comments disabled, the dataset consisted of 472 videos and 3018 comments. This was used to fine-tune the t5-small model. 

Here are some of the comments the model generated:

Video: "What's going on with Windows Laptops?" by Marques Brownlee

Comment: "It's not like a super high end gaming PC"[BREAK][BREAK]Me:


Video: "Traveling alone with my boyfriend for the first time" by HJ Evelyn

Comment: I love how Evelyn is doing a lot of her own things and she's so happy that she's getting back into shape.


Video: "Manchester United always score, but they don't anymore." by The Gary Neville Podcast

Comment: Manchester united need to look very hard above that very hard.

Overall, the model is great at using context from the video and generates some creative comments, but a lot of the time fails to produce sentences which make sense. It could be improved by using a narrower range of video topics as training data (for example, only training on fitness videos).

## How to run
Download the model from [Kaggle](https://www.kaggle.com/models/is0morphism/fine-tuned-t5)

Clone this repository and install the requirements.

To test the model on the sample data, run test.py. To test the model on an arbitrary video, run predict.py. You will need a Youtube API key to fetch the channel name and video title automatically, otherwise you can enter them in manually. 

