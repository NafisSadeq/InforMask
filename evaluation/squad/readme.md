# SQuAD evaluation Setup

The evaluation on SQuAD benchmark is done using the transformers library. The library need to be installed from source.

```
git clone https://github.com/huggingface/transformers
cd transformers
pip install .
pip install -r ./examples/requirements.txt
```

# Running 

```
cd examples/pytorch/question-answering/

python run_qa.py   --model_name_or_path nsadeq/InformBERT   --dataset_name squad   --do_train   --do_eval   --per_device_train_batch_size 12   --learning_rate 3e-5   --num_train_epochs 3 --save_total_limit 1   --max_seq_length 384   --doc_stride 128   --output_dir ./results/squad
python run_qa.py   --model_name_or_path nsadeq/InformBERT   --dataset_name squad_v2 --version_2_with_negative True   --do_train   --do_eval   --per_device_train_batch_size 12   --learning_rate 3e-5   --num_train_epochs 3 --save_total_limit 1   --max_seq_length 384   --doc_stride 128   --output_dir ./results/squad2
```