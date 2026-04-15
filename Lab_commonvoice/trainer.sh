#/home/gantumur/miniconda3/envs/env/bin/python /home/gantumur/Documents/DL/Lab_commonvoice/preprocess_translation.py --data commonvoice # deafult commonvoice and custom available

# /home/gantumur/miniconda3/envs/env/bin/python /home/gantumur/Documents/DL/Lab_commonvoice/train_whisper_medium.py --model_size medium --peft qlora --lr 3.5e-4 --batch_size 16 --eval_batch 4 --step 6000 --save_version 0.1 --train_data commonvoice
# /home/gantumur/miniconda3/envs/env/bin/python /home/gantumur/Documents/DL/Lab_commonvoice/train_whisper_medium.py --model_size medium --peft qlora --lr 3.5e-4 --batch_size 12 --eval_batch 4 --step 6000 --save_version 0.1 --train_data custom

/home/gantumur/miniconda3/envs/env/bin/python /home/gantumur/Documents/DL/Lab_commonvoice/train_whisper_medium.py --model_size small --peft fft --lr 3.5e-6 --batch_size 8 --eval_batch 8 --steps 8000  --save_version 0.2 --train_data commonvoice
/home/gantumur/miniconda3/envs/env/bin/python /home/gantumur/Documents/DL/Lab_commonvoice/train_whisper_medium.py --model_size small --peft fft --lr 1e-6 --batch_size 8 --eval_batch 8 --steps 4000  --save_version 0.1 --load_version 0.2 --train_data custom

# /home/gantumur/miniconda3/envs/env/bin/python /home/gantumur/Documents/DL/Lab_commonvoice/train_whisper_medium.py --model_size medium --peft fft --lr 3.5e-6 --batch_size 1 --eval_batch 1 --step 6000  --save_version 0.1 --train_data commonvoice
# /home/gantumur/miniconda3/envs/env/bin/python /home/gantumur/Documents/DL/Lab_commonvoice/train_whisper_medium.py --model_size medium --peft fft --lr 3.5e-6 --batch_size 1 --eval_batch 1 --step 6000  --save_version 0.1 --train_data custom



