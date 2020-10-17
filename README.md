This repo is meant to be run on the waterloo server with python-3.7.7

# Pruning

`git clone git@github.com:scribendi/Pruning.git`

`cd Pruning`

# Setting up the environment
`pip install -r requirements.txt`


## lets do the experiment in a folder called expt1

`mkdir expt1 ; cd expt1`

# Preprocessing 

## Now we are in expt1 directory. We want to train our model on 4million Scribendi data. First, we need to preprocess train and validation set.

## train data
`python ../utils/preprocess_data.py -s /data/faizy/spunkt/train.src  -t /data/faizy/spunkt/train.tgt -o data-train`

## validation data
`python ../utils/preprocess_data.py -s /data/faizy/spunkt/valid.src  -t /data/faizy/spunkt/valid.tgt  -o data-validation`


# Train
## current program use only one GPU card
## lets train a bert model with 12 encoder layers on gpu2. The trained model will be saved under bert-12 

`CUDA_VISIBLE_DEVICES=2 python ../train.py --train_set data-train --dev_set data-validation  --model_dir bert-12 --batch_size 32 --n_epoch 50 --transformer_model bert               --special_tokens_fix 0 --keep 12    --model_path . --max_len 60 --patience 10 --tune_bert 1 `  



# Inference
## We want to make inference on Scribendi test (small) data. The inference will be stored in a file named output-bert
### if the training is finished
`model_path="bert-12/model.th" ; vocab_path="bert-12/vocabulary/"`
### else use the back up model in the meantime
`model_path="backup-model/model.th" ; vocab_path="backup-model/vocabulary/"`

`CUDA_VISIBLE_DEVICES=2 python ../predict.py --model_path $model_path  --vocab_path $vocab_path --input_file /data/faizy/scribendidata/test-small.src  --output_file output-bert   --batch_size 1000 --transformer_model bert --special_tokens_fix 0 --keep 12`


# Metrics
## M2 Score
`bash ../get_m2score.bash output-bert`

## JFLEG Score
### predict on JFLEG data
`CUDA_VISIBLE_DEVICES=2 python ../predict.py --model_path $model_path --vocab_path $vocab_path --input_file /data/faizy/jfleg/test/test.src --output_file output-jfleg-bert --batch_size 1000 --transformer_model bert --special_tokens_fix 0 `

`bash ../get_jfleg_score.bash output-jfleg-bert ` 

## Errant Score
### first switch to errant environment
`source /data/faizy/errant_env/bin/activate`
` bash ../get_errant_score.bash output-bert`



