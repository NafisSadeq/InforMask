from datasets import load_dataset
from tqdm.auto import tqdm
import os

dataset = load_dataset("wikipedia","20220301.en")

text_data = []
file_count = 0

if (not os.path.exists("textcorpus")):
    os.makedirs("textcorpus")

for sample in tqdm(dataset['train']):
    
    sample = sample['text'].replace('\n', '')
    text_data.append(sample)
    
    if len(text_data) == 10_000:
        with open(f'textcorpus/text_{file_count}.txt', 'w', encoding='utf-8') as fp:
            fp.write('\n'.join(text_data))
        text_data = []
        file_count += 1
        
        
dataset = load_dataset("bookcorpus")

text_data = []

for sample in tqdm(dataset['train']):
    
    sample = sample['text'].replace('\n', '')
    text_data.append(sample)
    
    if len(text_data) == 10_000:
        with open(f'textcorpus/text_{file_count}.txt', 'w', encoding='utf-8') as fp:
            fp.write('\n'.join(text_data))
        text_data = []
        file_count += 1
