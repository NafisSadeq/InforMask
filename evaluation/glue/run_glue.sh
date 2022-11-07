export TASK_NAME=cola
python run_glue.py   --model_name_or_path nsadeq/InformBERT   --task_name $TASK_NAME   --do_train   --do_eval   --max_seq_length 128   --per_device_train_batch_size 32   --learning_rate 2e-5   --num_train_epochs 3 --save_total_limit 1  --output_dir results/$TASK_NAME/ --fp16 --overwrite_output_dir true
export TASK_NAME=sst2
python run_glue.py   --model_name_or_path nsadeq/InformBERT   --task_name $TASK_NAME   --do_train   --do_eval   --max_seq_length 128   --per_device_train_batch_size 32   --learning_rate 2e-5  --num_train_epochs 3 --save_total_limit 1  --output_dir results/$TASK_NAME/ --fp16 --overwrite_output_dir true 
export TASK_NAME=mrpc
python run_glue.py   --model_name_or_path nsadeq/InformBERT   --task_name $TASK_NAME   --do_train   --do_eval   --max_seq_length 128   --per_device_train_batch_size 32   --learning_rate 2e-5   --num_train_epochs 3 --save_total_limit 1  --output_dir results/$TASK_NAME/ --fp16 --overwrite_output_dir true 
export TASK_NAME=stsb
python run_glue.py   --model_name_or_path nsadeq/InformBERT   --task_name $TASK_NAME   --do_train   --do_eval   --max_seq_length 128   --per_device_train_batch_size 32   --learning_rate 2e-5   --num_train_epochs 3 --save_total_limit 1  --output_dir results/$TASK_NAME/ --fp16 --overwrite_output_dir true 
export TASK_NAME=qqp
python run_glue.py   --model_name_or_path nsadeq/InformBERT   --task_name $TASK_NAME   --do_train   --do_eval   --max_seq_length 128   --per_device_train_batch_size 32   --learning_rate 2e-5   --num_train_epochs 3 --save_total_limit 1  --output_dir results/$TASK_NAME/ --fp16 --overwrite_output_dir true 
export TASK_NAME=mnli
python run_glue.py   --model_name_or_path nsadeq/InformBERT   --task_name $TASK_NAME   --do_train   --do_eval   --max_seq_length 128   --per_device_train_batch_size 32   --learning_rate 2e-5   --num_train_epochs 3 --save_total_limit 1  --output_dir results/$TASK_NAME/ --fp16 --overwrite_output_dir true 
export TASK_NAME=qnli
python run_glue.py   --model_name_or_path nsadeq/InformBERT   --task_name $TASK_NAME   --do_train   --do_eval   --max_seq_length 128   --per_device_train_batch_size 32   --learning_rate 2e-5   --num_train_epochs 3 --save_total_limit 1  --output_dir results/$TASK_NAME/ --fp16 --overwrite_output_dir true 
export TASK_NAME=rte
python run_glue.py   --model_name_or_path nsadeq/InformBERT   --task_name $TASK_NAME   --do_train   --do_eval   --max_seq_length 128   --per_device_train_batch_size 32   --learning_rate 2e-5   --num_train_epochs 3 --save_total_limit 1  --output_dir results/$TASK_NAME/ --fp16 --overwrite_output_dir true 
export TASK_NAME=wnli
python run_glue.py   --model_name_or_path nsadeq/InformBERT   --task_name $TASK_NAME   --do_train   --do_eval   --max_seq_length 128   --per_device_train_batch_size 32   --learning_rate 2e-5   --num_train_epochs 3 --save_total_limit 1  --output_dir results/$TASK_NAME/ --fp16 --overwrite_output_dir true
