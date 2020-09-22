

if [ $# -lt 1 ]
then
    echo "bash file.bash pred-text"
    exit 1
fi

pred=$1

errant_parallel -orig /data/faizy/scribendidata/test-small.src \
-cor $pred  \
-out $pred-errant.m2

#errant_compare -hyp $pred-errant.m2  \
#-ref /data/faizy/errant/conll-2014-test-errant.m2  -cat 1
#errant_compare -hyp $pred-errant.m2  \
#-ref /data/faizy/errant/conll-2014-test-errant.m2  -cat 1 -ds
#errant_compare -hyp $pred-errant.m2  \
#-ref /data/faizy/errant/conll-2014-test-errant.m2  -cat 1 -dt
#
#errant_compare -hyp $pred-errant.m2  \
#-ref /data/faizy/errant/conll-2014-test-errant.m2  -cat 3
#errant_compare -hyp $pred-errant.m2  \
#-ref /data/faizy/errant/conll-2014-test-errant.m2  -cat 3 -ds
#errant_compare -hyp $pred-errant.m2  \
#-ref /data/faizy/errant/conll-2014-test-errant.m2  -cat 3 -dt

# scribendi test small
errant_compare -hyp $pred-errant.m2  \
-ref /data/faizy/scribendidata/test-small-errant.m2  -cat 1
errant_compare -hyp $pred-errant.m2  \
-ref /data/faizy/scribendidata/test-small-errant.m2 -cat 1 -ds
errant_compare -hyp $pred-errant.m2  \
-ref /data/faizy/scribendidata/test-small-errant.m2  -cat 1 -dt

errant_compare -hyp $pred-errant.m2  \
-ref /data/faizy/scribendidata/test-small-errant.m2  -cat 3
errant_compare -hyp $pred-errant.m2  \
-ref /data/faizy/scribendidata/test-small-errant.m2  -cat 3 -ds
errant_compare -hyp $pred-errant.m2  \
-ref /data/faizy/scribendidata/test-small-errant.m2 -cat 3 -dt

