from pathlib import Path
import torch
from torch.utils import data
from transformers import BertTokenizerFast
import random
from scipy.stats import geom
import json,gzip

class Dataset(data.Dataset):
    
    def __init__(self, file_path,recursive=False,vocab_folder="../vocab",doc_len=512,mask_perc=0.15,masking_type="informask"):
        
        super().__init__()
        
        self.file_info = []
        self.encodings={}
        self.curr_file_index=-1
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.CLS_id=self.tokenizer.convert_tokens_to_ids('[CLS]')
        self.PAD_id=self.tokenizer.convert_tokens_to_ids('[PAD]')
        self.SEP_id=self.tokenizer.convert_tokens_to_ids('[SEP]')
        self.MASK_id=self.tokenizer.convert_tokens_to_ids('[MASK]')
        self.masking_type=masking_type
        self.mask_perc = mask_perc
        self.doc_len=doc_len
        
        self.pmi_dict=None
        self.id_to_prob_dict=[0.0]*31000
        self.num_pmi_span=45
        
        self.num_mask_span=17
        self.p_geometric=0.2
        
        p = Path(file_path)
        
        if(self.masking_type=="pmi_masking"):
            # with open(vocab_folder+"/idx_pmi_dict.json.gz",'r') as file:
            #     self.pmi_dict=json.load(file)
                
            with gzip.open(vocab_folder+"/idx_pmi_dict.json.gz",'r') as file:
                self.pmi_dict = json.loads(file.read().decode('utf-8'))
        
        if(self.masking_type=="informask"):
            with open(vocab_folder+"/id_mask_prob.json",'r') as file:
                id_to_prob_dict=json.load(file)
                for key,value in id_to_prob_dict.items():
                    self.id_to_prob_dict[int(key)]=float(value)
        
        print("Masking Type Chosen: ",self.masking_type)
        
        assert(p.is_dir())
        
        if recursive:
            files = sorted(p.glob('**/*.txt'))
        else:
            files = sorted(p.glob('*.txt'))
        if len(files) < 1:
            raise RuntimeError('No dataset file found')

        for dataset_fp in files:
            self.file_info.append(dataset_fp)
            
        random.shuffle(self.file_info)
            
        self._load_data(file_index=0)
              
    def __getitem__(self, i):
        
        file_index=i//self.encodings['input_ids'].shape[0]
        item_index=i%self.encodings['input_ids'].shape[0]
        
        if(file_index!=self.curr_file_index):
            self._load_data(file_index=file_index)
            
        encoded_dict={key: tensor[item_index] for key, tensor in self.encodings.items()}
        
        if(self.masking_type=="informask"):

            input_ids=encoded_dict["labels"].clone()
            rand = torch.rand(input_ids.shape)
            indices=input_ids.tolist()
            values=[0.0]*input_ids.shape[0]

            for i in range(len(indices)):

                wid=indices[i]
                values[i]=self.id_to_prob_dict[wid]

            probs = torch.tensor(values, dtype=torch.float)

            mask_arr = (rand < probs) * (input_ids != self.PAD_id) * (input_ids != self.CLS_id) * (input_ids != self.SEP_id)

            selection = torch.flatten(mask_arr.nonzero()).tolist()
            input_ids[selection] = self.MASK_id 

            encoded_dict["input_ids"]=input_ids
            
        elif(self.masking_type=="span"):
            
            span_size_arr = geom.rvs(self.p_geometric, size=self.num_mask_span)
            input_ids=encoded_dict["labels"].clone()
            mask_arr = (input_ids != self.PAD_id) * (input_ids != self.CLS_id) * (input_ids != self.SEP_id)
            span_start_indices=random.sample(range(input_ids.shape[0]),self.num_mask_span)
            mask_indices=[]
            for i in range(span_size_arr.shape[0]):
                
                start_index=span_start_indices[i]
                
                for j in range(span_size_arr[i]):
                    
                    k=start_index+j
                    
                    if ((start_index+j)<input_ids.shape[0]
                        and mask_arr[k]):
                        mask_indices.append(k)
                        
            input_ids[mask_indices] = self.MASK_id           
            encoded_dict["input_ids"]=input_ids
            
        elif(self.masking_type=="pmi_masking"):
            
            input_ids=encoded_dict["labels"].clone()
            span_start_indices=random.sample(range(input_ids.shape[0]),self.num_pmi_span)
            mask_arr = (input_ids != self.PAD_id) * (input_ids != self.CLS_id) * (input_ids != self.SEP_id)
            mask_indices=[]
            
            for i in span_start_indices:
                
                if(not mask_arr[i]):
                    continue
                    
                mask_indices.append(i)
                
                j=i
                while (j<(input_ids.shape[0]-1)
                    and mask_arr[j+1] 
                    and (str(input_ids[j].item()) in self.pmi_dict) 
                    and (str(input_ids[j+1].item()) in self.pmi_dict[str(input_ids[j].item())])):
                    mask_indices.append(j+1)
                    j=j+1
                j=i
                while ( (j>1)
                and mask_arr[j-1]
                and (str(input_ids[j-1].item()) in self.pmi_dict) 
                and (str(input_ids[j].item()) in self.pmi_dict[str(input_ids[j-1].item())])):
                    mask_indices.append(j-1)
                    j=j-1
            
            input_ids[mask_indices] = self.MASK_id  
            encoded_dict["input_ids"]=input_ids
            
        else:
            
            input_ids=encoded_dict["labels"].clone()
            rand = torch.rand(input_ids.shape)
            mask_arr = (rand<self.mask_perc)*(input_ids!=self.PAD_id)*(input_ids!=self.CLS_id)*(input_ids!=self.SEP_id)

            selection = torch.flatten(mask_arr.nonzero()).tolist()
            input_ids[selection] = self.MASK_id 

            encoded_dict["input_ids"]=input_ids

        return encoded_dict

    def __len__(self):

        return self.encodings['input_ids'].shape[0]*len(self.file_info)
    
    def _load_data(self,file_index):

        lines=[]

        with open(self.file_info[file_index],'r') as file:

            for line in file:
                lines.append(line.strip())

        encoding_list=self.tokenizer(lines, max_length=self.doc_len, padding='max_length', truncation=True)

        labels = torch.tensor([x for x in encoding_list["input_ids"]])
        mask = torch.tensor([x for x in encoding_list["attention_mask"]])
        self.encodings["input_ids"]=labels
        self.encodings["labels"]=labels.detach().clone()
        self.encodings["attention_mask"]=mask

        self.curr_file_index=file_index 
