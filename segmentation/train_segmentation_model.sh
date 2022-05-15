echo *** make sure the individual lead pictures have been generated from PTB-V using dataset_generation/data_augmentation.ipynb! ***

python3 main.py --dataset=ptb_v --input_shape="((None,None,3))" --num_epochs=100 --split=0.98 --model=UNetTF --experiment_name=UNetTF --checkpoint_interval=200