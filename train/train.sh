DATA_DIR=/home/yxchng/refined-ms1m-112x96
DATA_LIST=/home/yxchng/temp.txt

python3 -u train.py --data_dir=$DATA_DIR --data_list=$DATA_LIST 2>&1 | tee sphereface.log
