if [ $# -lt 1 ]
then
    echo "bash file.bash pred-text"
    exit 1
fi

pred=$1

python /data/faizy/jfleg/eval/gleu.py -r /data/faizy/jfleg/test/test.ref[0-3] \
-s /data/faizy/jfleg/test/test.src --hyp $pred
