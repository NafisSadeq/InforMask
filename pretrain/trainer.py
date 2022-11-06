import torch
from torch.utils import data
from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader,Dataset
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data.distributed import DistributedSampler
import json


class Trainer(Trainer):
    
    def get_train_dataloader(self) -> DataLoader:
        
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        train_data = self.train_dataset
        
        sampler=DistributedSampler(
                    self.train_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    shuffle=False,
                    drop_last=True
                )
        
        return torch.utils.data.DataLoader(train_data, batch_size=self.args.train_batch_size, sampler=sampler)
    
    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        
        return torch.utils.data.DataLoader(eval_dataset, batch_size=self.args.eval_batch_size, shuffle=False)
    
    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        
        if is_datasets_available() and isinstance(test_dataset, datasets.Dataset):
            test_dataset = self._remove_unused_columns(test_dataset, description="test")
            
        return torch.utils.data.DataLoader(test_dataset, batch_size=self.args.eval_batch_size, shuffle=False)