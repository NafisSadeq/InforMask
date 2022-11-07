from tqdm.auto import tqdm
import torch
from torch.utils import data
from transformers import BertTokenizerFast
from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
from transformers import pipeline
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW
import json
from pathlib import Path
import sys
import os

tokenizer = BertTokenizerFast.from_pretrained('nsadeq/InformBERT')

file_paths = [str(x) for x in Path('data').glob('*/*.jsonl')]
null=None 

vocab_list=[]

with open("../../vocab/vocab_merged.txt",'r') as file:

    for line in file:
        vocab_list.append(line.strip())
    
vocab_set=set(vocab_list)

file_paths.sort()

output_path="results"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
fill = pipeline('fill-mask', model='nsadeq/InformBERT', tokenizer=tokenizer,top_k=30,device=0)
print(device)
    

if not os.path.exists(output_path):
    os.makedirs(output_path)

output_path=output_path+"/eval_results.csv"

with open(output_path,'w') as file:
    file.write("x,x,x,x\n")
    

for file_path in file_paths:

    result=[]

    print("File path:",file_path)

    result.append(file_path.split('/')[-2])
    result.append(file_path.split('/')[-1])

    test_dict_list=[]

    with open(file_path,'r') as file:

        lines=file.readlines()

    for line in lines:

        test_dict=eval(line.strip())
        test_dict_list.append(test_dict)


    print("Number of Samples:",len(test_dict_list))

    reciprocal_rank_list=[]

    for test_dict in tqdm(test_dict_list):

        if("TREx" in file_path):
            masked_sent=test_dict["evidences"][0]['masked_sentence']
            test_label=test_dict["evidences"][0]['obj_surface'].lower()
        else:
            masked_sent=test_dict['masked_sentences'][0]
            test_label=test_dict['obj_label'].lower()    

        if("[MASK]" not in masked_sent or len(tokenizer.tokenize(masked_sent))>500):
            continue

        test_label_list=test_label.split(",")
        test_label_list=[x.strip() for x in test_label_list]

        preds=fill(masked_sent)
        flag=True

        preds_filtered=[]

        for p in preds:
            if("token_str" in p):
                pred_label=p['token_str'].replace(' ','').lower()

                if(pred_label in vocab_set or pred_label in test_label_list):
                    preds_filtered.append(pred_label)

        for i,pred_label in enumerate(preds_filtered):

            if(pred_label in test_label_list):
                reciprocal_rank_list.append(1/(i+1)) 
                flag=False
                break
        if(flag):
            reciprocal_rank_list.append(0) 

    print("Processed Samples:",len(reciprocal_rank_list))

    result.append(str(len(reciprocal_rank_list)))

    MRR=sum(reciprocal_rank_list)/len(reciprocal_rank_list)

    print("MRR:",MRR)
    print()

    result.append(str(round(MRR,3)))

    result=",".join(result)

    with open(output_path,'a') as file:
        file.write(result+"\n")
