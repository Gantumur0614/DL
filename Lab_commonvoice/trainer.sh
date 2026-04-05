conda activate env 

# /home/gantumur/miniconda3/envs/env/bin/python /home/gantumur/Documents/DL/Lab_commonvoice/train_whisper_medium.py --model_size medium --peft qlora --lr 3.5e-4 --batch_size 16 --eval_batch 4 --step 6000 --save_version 0.1 --train_data commonvoice
# /home/gantumur/miniconda3/envs/env/bin/python /home/gantumur/Documents/DL/Lab_commonvoice/train_whisper_medium.py --model_size medium --peft qlora --lr 3.5e-4 --batch_size 12 --eval_batch 4 --step 6000 --save_version 0.1 --train_data custom

/home/gantumur/miniconda3/envs/env/bin/python /home/gantumur/Documents/DL/Lab_commonvoice/train_whisper_medium.py --model_size medium --peft lora --lr 3.5e-5 --batch_size 8 --eval_batch 2 --steps 6000  --save_version 0.1 --train_data commonvoice
/home/gantumur/miniconda3/envs/env/bin/python /home/gantumur/Documents/DL/Lab_commonvoice/train_whisper_medium.py --model_size medium --peft lora --lr 3.5e-5 --batch_size 8 --eval_batch 2 --steps 6000  --save_version 0.1 --train_data custom

# /home/gantumur/miniconda3/envs/env/bin/python /home/gantumur/Documents/DL/Lab_commonvoice/train_whisper_medium.py --model_size medium --peft fft --lr 3.5e-6 --batch_size 1 --eval_batch 1 --step 6000  --save_version 0.1 --train_data commonvoice
# /home/gantumur/miniconda3/envs/env/bin/python /home/gantumur/Documents/DL/Lab_commonvoice/train_whisper_medium.py --model_size medium --peft fft --lr 3.5e-6 --batch_size 1 --eval_batch 1 --step 6000  --save_version 0.1 --train_data custom




