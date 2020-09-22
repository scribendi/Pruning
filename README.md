This repo is meant to be run on the waterloo server

# Pruning

`git clone git@github.com:scribendi/Pruning.git`

`cd Pruning`

# Setting up the environment
`source /data/faizy/miniconda3/bin/activate`

`conda activate tf-gpu`

## lets do the experiment in a folder called expt1

`mkdir expt1 ; cd expt1`

# Preprocessing 

## Now we are in expt1 directory. We want to train our model on 4million Scribendi data. First, we need to preprocess train and validation set.

## train
`python ../utils/preprocess_data.py -s /data/faizy/scribendidata/src-train.txt -t /data/faizy/scribendidata/targ-train.txt -o data-train`

## validation
`python ../utils/preprocess_data.py -s /data/faizy/scribendidata/src-validation.txt -t /data/faizy/scribendidata/targ-validation.txt -o data-validation`


# Train
## current program use only one GPU card
## lets train a bert model on gpu3. The trained model will be saved under bert-model 

`CUDA_VISIBLE_DEVICES=3 python ../custom_train.py --train_set data-train --dev_set data-validation --model_dir bert-model --batch_size 256 --n_epoch 50 --transformer_model bert  --special_tokens_fix 0 `  



# Inference
## We want to make inference on Scribendi test (small) data. The inference will be stored in a file named output-bert
`CUDA_VISIBLE_DEVICES=3 python ../custom_predict.py --model_path  --vocab_path --input_file /data/faizy/scribendidata/test-small.src --output_file output-bert --batch_size 1000 --transformer_model bert --special_tokens_fix 0 `


# Metrics
## M2 Score

## JFLEG Score
### predict

## Errant Score


# Pruning


