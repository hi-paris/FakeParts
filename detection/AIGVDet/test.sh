#python test.py -fop "data/test/T2V/hotshot" -mop "checkpoints/optical_aug.pth" -for "data/test/original/T2V/hotshot" -mor "checkpoints/original_aug.pth" -e "data/results/T2V/hotshot.csv" -ef "data/results/frame/T2V/hotshot.csv" -t 0.5

python test.py -fop "data/train/testset_2" \
               -mop "/data/parietal/store3/work/gbrison/gen/detectors/AIGVDet/data/exp/train_testset_2/ckpt/model_epoch_latest.pth" \
               -for "data/train/testset_2" \
               -mor "/data/parietal/store3/work/gbrison/gen/detectors/AIGVDet/data/exp/train_testset_2/ckpt/model_epoch_latest.pth" \
               -e "data/results/T2V/hotshot.csv" \
               -ef "data/results/frame/T2V/hotshot.csv" \
               -t 0.5
