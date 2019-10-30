# CUDA_VISIBLE_DEVICES=3 python trainval_net.py --dataset pascal_voc --net vgg16\
#                    --bs 1 --nw 0 \
#                    --lr 0.001\
#                    --lr_decay_step 5 --epochs 6\
#                    --frozen_status 1\
#                    --flag lowshot1\
#                    --use_tfb\
#                    --cuda
                   
#                 #    --r True --checkepoch 1 --checkepoch 7 --checkpoint 10021\


# CUDA_VISIBLE_DEVICES=3 python trainval_net.py --dataset pascal_voc --net vgg16\
#                    --bs 1 --nw 0 \
#                    --lr 0.0005\
#                    --lr_decay_step 10 --epochs 12\
#                    --r True --checkepoch 1 --checkepoch 6 --checkpoint 1801\
#                    --frozen_status 4\
#                    --flag lowshot1\
#                    --use_tfb\
#                    --cuda\
#                    --re_class
                   


CUDA_VISIBLE_DEVICES=3 python test_net.py --dataset pascal_voc --net vgg16 \
                   --checksession 1 --checkepoch 11 --checkpoint 1801 \
                   --flag lowshot1\
                   --re_class\
                   --cuda

CUDA_VISIBLE_DEVICES=3 python test_net.py --dataset pascal_voc --net vgg16 \
                   --checksession 1 --checkepoch 12 --checkpoint 1801 \
                   --flag lowshot1\
                   --re_class\
                   --cuda