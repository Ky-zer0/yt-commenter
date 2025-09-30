from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = T5ForConditionalGeneration.from_pretrained('./fine-tuned-t5-v8').to(device)
tokenizer = T5Tokenizer.from_pretrained('./fine-tuned-t5-v8')

def contextualise(title,channel_name,transcript):
    return f"Title: {title} | Channel: {channel_name} | Transcript: {transcript}"

def generate_comment(contextual_input):
    inputs = tokenizer(contextual_input, return_tensors='pt', max_length=512, truncation=True).to(device)
    outputs = model.generate(inputs['input_ids'], max_length=128, num_beams=5,repetition_penalty=3.0, early_stopping=True, do_sample=True,temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

f=open("test_data.txt","r", encoding="utf-8")
data=f.readlines()
for i in data:
    generated_comment = generate_comment(i)
    print(generated_comment)