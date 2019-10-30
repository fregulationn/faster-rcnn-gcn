# CUDA_VISIBLE_DEVICES=0,1 python trainval_net.py --dataset pascal_voc --net vgg16\
#                    --bs 2 --nw 2 \
#                    --lr 0.002\
#                    --lr_decay_step 5 --epochs 6\
#                    --frozen_status 1\
#                    --flag lowshot\
#                    --use_tfb\
#                    --cuda\
#                    --mGPUs
#                 #    --r True --checkepoch 1 --checkepoch 7 --checkpoint 10021\


# CUDA_VISIBLE_DEVICES=0 python trainval_net.py --dataset pascal_voc --net vgg16\
#                    --bs 1 --nw 1 \
#                    --lr 0.0005\
#                    --lr_decay_step 10 --epochs 12\
#                    --r True --checkepoch 1 --checkepoch 6 --checkpoint 900\
#                    --frozen_status 4\
#                    --flag lowshot\
#                    --use_tfb\
#                    --cuda\
#                    --re_class


CUDA_VISIBLE_DEVICES=0 python test_net.py --dataset pascal_voc --net vgg16 \
                   --checksession 1 --checkepoch 6 --checkpoint 900 \
                   --flag lowshot\
                   --cuda

CUDA_VISIBLE_DEVICES=0 python test_net.py --dataset pascal_voc --net vgg16 \
                   --checksession 1 --checkepoch 12 --checkpoint 1801 \
                   --flag lowshot\
                   --re_class\
                   --cuda