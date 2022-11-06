from pathlib import Path
from transformers import BertTokenizerFast
from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
from transformers import Trainer, TrainingArguments
from dataset import Dataset
from transformers import TrainingArguments
from trainer import Trainer
import os
import argparse

parser = argparse.ArgumentParser(description='RoBERTa style training')
parser.add_argument("-mt",'--masking_type', type=str, default="informask",choices=["informask", "random", "span","pmi_masking"])
parser.add_argument("--local_rank", type=int, default=-1)
args = parser.parse_args()

masking_type=args.masking_type 

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

if (not os.path.exists("models")):
    os.makedirs("models")

model_path = "./models/"+masking_type
    
train_data = Dataset("textcorpus",doc_len=512,mask_perc=0.15,masking_type=args.masking_type)

config = RobertaConfig(
    vocab_size=50265,
    max_position_embeddings=514,
    hidden_size=768,
    num_attention_heads=12,
    num_hidden_layers=12,
    type_vocab_size=2
)

model = RobertaForMaskedLM(config=config)

print("Num param: ",model.num_parameters())

training_args = TrainingArguments(
    output_dir=model_path,
    overwrite_output_dir=True,
    num_train_epochs=40,
    per_device_train_batch_size=16,
    save_strategy="epoch",
    prediction_loss_only=True,
    local_rank=args.local_rank,
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
)

trainer.train()