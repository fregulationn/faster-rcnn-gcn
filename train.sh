python trainval_net.py --dataset pascal_voc --net res101 \
                   --bs 1 --nw 0 \
                   --lr 0.001 \
                   --lr_decay_step 12 --epochs 14\
                   --frozen_status 3\
                   --save_dir models_frozen\
                   --o adam \
                   --r True --checkepoch 1 --checkepoch 7 --checkpoint 10021\
                   --use_tfb\
                   --cuda

python test_net.py --dataset pascal_voc --net res101 \
                   --checksession 1 --checkepoch 7 --checkpoint 10021 \
                   --load_dir models_frozen\
                   --save_name forzen_og\
                   --cuda

python trainval_net.py --dataset pascal_voc --net res101 \
                   --bs 1 --nw 0 \
                   --lr 0.001 \
                   --lr_decay_step 12 --epochs 14\
                   --frozen_status 1\
                   --o adam \
                   --r True --checkepoch 1 --checkepoch 7 --checkpoint 10021\
                   --use_tfb\
                   --cuda

python test_net.py --dataset pascal_voc --net res101 \
                   --checksession 1 --checkepoch 7 --checkpoint 10021 \
                   --save_name og\
                   --cuda