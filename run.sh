DATA_DIR="/mnt/storage1/AV-NeRF/release"
LOG_DIR="./logs_test" # --eval / sa: save audios / mask: mask + rgb visual feature / depth_mask: depth + mask / logs_mask_2_ras:   / logs_depth_mask_2_ras: / logs_GT: upper bound 확인용
EXP_NAME="reproduce"
METHOD="anerf"
for i in 7
do
    echo $i
    python main.py --data-root ${DATA_DIR}/$i/ --log-dir ${LOG_DIR}/$i/audio_output/ --output-dir $METHOD/$EXP_NAME/ --conv --lr 5e-4 --max-epoch 100
done

python eval.py --log-dir ${LOG_DIR}/ --output-dir $METHOD/$EXP_NAME/