CUDA_VISIBLE_DEVICES=0 python trainval_net.py --dataset pascal_voc --net res101 \
                   --bs 1 --nw 0 \
                   --lr 0.005\
                   --lr_decay_step 12 --epochs 14\
                   --r True --checkepoch 1 --checkepoch 7 --checkpoint 10021\
                   --frozen_status 4\
                   --flag bg\
                   --use_tfb\
                   --cuda
                #    --r True --checkepoch 1 --checkepoch 7 --checkpoint 10021\

# CUDA_VISIBLE_DEVICES=0 python test_net.py --dataset pascal_voc --net res101 \
#                    --checksession 1 --checkepoch 8 --checkpoint 10021 \
#                    --flag o\
#                    --cuda

# CUDA_VISIBLE_DEVICES=0 python trainval_net.py --dataset pascal_voc --net res101 \
#                    --bs 1 --nw 0 \
#                    --lr 0.01\
#                    --lr_decay_step 12 --epochs 14\
#                    --frozen_status 4\
#                    --o adam\
#                    --flag adam\
#                    --r True --checkepoch 1 --checkepoch 7 --checkpoint 10021\
#                    --use_tfb\
#                    --cuda

# python trainval_net.py --dataset pascal_voc --net res101 \
#                    --bs 1 --nw 0 \
#                    --lr 0.0005\
#                    --lr_decay_step 12 --epochs 14\
#                    --frozen_status 3\
#                    --flag sgd3\
#                    --r True --checkepoch 1 --checkepoch 7 --checkpoint 10021\
#                    --use_tfb\
#                    --cuda

# python trainval_net.py --dataset pascal_voc --net res101 \
#                    --bs 1 --nw 0 \
#                    --lr 0.001\
#                    --lr_decay_step 12 --epochs 14\
#                    --flag satatus1\
#                    --o adam \
#                    --r True --checkepoch 1 --checkepoch 7 --checkpoint 10021\
#                    --use_tfb\
#                    --cuda

# python test_net.py --dataset pascal_voc --net res101 \
#                    --checksession 1 --checkepoch 7 --checkpoint 10021 \
#                    --load_dir models_frozen\
#                    --save_name forzen_og\
#                    --cuda


# python test_net.py --dataset pascal_voc --net res101 \
#                    --checksession 1 --checkepoch 7 --checkpoint 10021 \
#                    --save_name og\
#                    --cuda