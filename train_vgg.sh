CUDA_VISIBLE_DEVICES=0 python trainval_net.py --dataset pascal_voc --net vgg16\
                   --bs 1 --nw 0 \
                   --lr 0.005\
                   --lr_decay_step 11 --epochs 13\
                   --r True --checkepoch 1 --checkepoch 6 --checkpoint 10021\
                   --frozen_status 4\
                   --flag vgg_bg\
                   --use_tfb\
                   --cuda
                #    --r True --checkepoch 1 --checkepoch 7 --checkpoint 10021\

CUDA_VISIBLE_DEVICES=0 python test_net.py --dataset pascal_voc --net vgg16 \
                   --checksession 1 --checkepoch 14 --checkpoint 10021 \
                   --flag vgg_bg\
                   --cuda