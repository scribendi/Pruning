
if [ $# -lt 1 ]
then
    echo "bash file.bash pred-text"
    exit 1
fi

pred=$1

# tokenize
python2 /data/faizy/m2scorer-master/scripts/Tokenizer.py < $pred  > $pred.tok

## compute m2score on conll-2014
#time python2 /data/faizy/m2scorer-master/scripts/m2scorer.py \
#    --max_unchanged_words 30 \
#    $pred.tok \
#    /data/faizy/conll14st-test-data/noalt/official-2014.combined.m2


# compute m2score on scribendi-test
time python2 /data/faizy/m2scorer-master/scripts/m2scorer.py \
    --max_unchanged_words 30 \
    $pred.tok \
    /data/faizy/scribendidata/test-small.m2

