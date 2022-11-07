# InforMask
Code for EMNLP 2022 paper: [InforMask: Unsupervised Informative Masking for Language Model Pretraining](https://arxiv.org/abs/2210.11771). Our model is pretrained using a variable masking strategy, where informative tokens are masked more frequently compared to other tokens. InformBERT outperforms random masking-based pretrained models on the factual recall benchmark LAMA and extractive question answering benchmark SQuAD.

## Model Download

You can directly download the use the model from Hugging Face repository [model link](https://huggingface.co/nsadeq/InformBERT). The evaluation code in this repository will also directly download the model from Hugging Face. Example usage is shown below:

```
from transformers import BertTokenizer, AutoModel
tokenizer = BertTokenizer.from_pretrained("nsadeq/InformBERT")
model = AutoModel.from_pretrained("nsadeq/InformBERT")

from transformers import pipeline
unmasker = pipeline('fill-mask', model='nsadeq/InformBERT',tokenizer=tokenizer)
unmasker("SpeedWeek is an American television program on [MASK].")
```

## Setup

```
pip install numpy pandas matplotlib seaborn scikit-learn torch tensorboard transformers datasets apache_beam
```


## Pretraining

We use 16 V100 GPUs for training, with per device batch size set to 16. If you use a different setting, update the batch size accordingly. 

```
cd pretrain
python download_corpus.py
python -m torch.distributed.launch --nproc_per_node=16 train.py -mt informask

```

You can try different masking strategies by using the -mt argument. Other available masking strategies are random, span and pmi_masking. Details regarding the impact of masking strategies can be found in the [paper](https://arxiv.org/abs/2210.11771).

## Evaluation

We perform evaluation on LAMA, SQuAD, and GLUE. Check in the corresponding folder under the evaluation directory for detailed instructions.

## Citation
If you use the proposed approach, please cite the following work.

```
@misc{https://doi.org/10.48550/arxiv.2210.11771,
  doi = {10.48550/ARXIV.2210.11771},
  url = {https://arxiv.org/abs/2210.11771},
  author = {Sadeq, Nafis and Xu, Canwen and McAuley, Julian},
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences}, 
  title = {InforMask: Unsupervised Informative Masking for Language Model Pretraining},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```


