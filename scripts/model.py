#training the model
import sqlite3
import pandas as pd

conn = sqlite3.connect('video_data.db')
transcripts_df = pd.read_sql_query(
    "SELECT video_id, title, channel, transcript_summarised FROM transcripts", conn
)
comments_df = pd.read_sql_query("SELECT video_id, comment FROM comments", conn)
conn.close()
merged_df = pd.merge(comments_df, transcripts_df, on='video_id')

#adding title and channel name to the front of each transcript
merged_df['contextual_transcript'] = ("Title: "+merged_df['title']+" | Channel: "+merged_df['channel']+" | Transcript: "+merged_df['transcript_summarised'])

train_data = list(zip(merged_df['contextual_transcript'], merged_df['comment']))
print(len(train_data))

from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(merged_df, test_size=0.1)

train_data = list(zip(train_df['contextual_transcript'], train_df['comment']))
val_data = list(zip(val_df['contextual_transcript'], val_df['comment']))

from transformers import T5Tokenizer

model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    inputs = tokenizer(examples['input_text'], max_length=512, truncation=True, padding="max_length")
    targets = tokenizer(examples['target_text'], max_length=128, truncation=True, padding="max_length")
    inputs['labels'] = targets['input_ids']
    return inputs

from datasets import Dataset, DatasetDict

train_dataset = Dataset.from_pandas(pd.DataFrame(train_data, columns=['input_text', 'target_text']))
val_dataset = Dataset.from_pandas(pd.DataFrame(val_data, columns=['input_text', 'target_text']))

#tokenize datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

from transformers import T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq

print("datasets prepared")
#load model
model = T5ForConditionalGeneration.from_pretrained(model_name)

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=40,
    weight_decay=0.01,
    save_total_limit=1,
    fp16=True
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

model.save_pretrained('./fine-tuned-t5-v9')
tokenizer.save_pretrained('./fine-tuned-t5-v9')
