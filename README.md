This repo should be run with python-3.7.7

# Pruning

`git clone git@github.com:scribendi/Pruning.git`

`cd Pruning`

# Setting up the environment
`pip install -r requirements.txt`


## lets do the experiment in a folder called expt1

`mkdir expt1 ; cd expt1`

# Preprocessing 

## Now we are in expt1 directory. First, we need to preprocess train and validation set.

## train data
`python ../utils/preprocess_data.py -s Path_To_Erroneous_Train_Data -t Path_To_Correct_Train_Data  -o data-train`


## validation data
`python ../utils/preprocess_data.py -s Path_To_Erroneous_Validation_Data  -t Path_To_Correct_Validation_Data  -o data-validation`


# Train
## current program use only one GPU card
## lets train a bert model with 12 encoder layers on gpu2. The trained model will be saved under bert-12 

`CUDA_VISIBLE_DEVICES=2 python ../train.py --train_set data-train --dev_set data-validation  --model_dir bert-12 --batch_size 32 --n_epoch 50 --transformer_model bert               --special_tokens_fix 0 --keep 12    --model_path . --max_len 60 --patience 10 --tune_bert 1 `  



# Inference
## We want to make inference on the test data. The inference will be stored in a file named output-bert

`CUDA_VISIBLE_DEVICES=2 python ../predict.py --model_path Path_To_Model  --vocab_path Path_To_Vocabulary --input_file Path_To_Test_Data  --output_file output-bert   --batch_size 1000 --transformer_model bert --special_tokens_fix 0 --keep 12`




