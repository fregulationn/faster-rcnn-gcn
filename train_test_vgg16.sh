#!/usr/bin/env sh

GPU_ID=0
BATCH_SIZE=1
WORKER_NUMBER=1
LEARNING_RATE=0.001

TRAIN=0
TEST=1

if  [ $TRAIN = 1 ]; then
	CUDA_VISIBLE_DEVICES=$GPU_ID python trainval_net.py \
                   --dataset pascal_voc --net vgg16 \
                   --bs $BATCH_SIZE --nw $WORKER_NUMBER \
                   --cuda \
                   --use_tfb\
                   --lr_decay_step 10\
                   --r True --checkepoch 1 --checkepoch 6 --checkpoint 10021\
                   --epochs 12
fi

if [ $TEST = 1 ]; then
	CUDA_VISIBLE_DEVICES=$GPU_ID python test_net.py --dataset pascal_voc --net vgg16 \
       --checksession 1 --checkepoch 11 --checkpoint 10021 --cuda
fi

