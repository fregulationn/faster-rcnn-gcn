# python test_net.py --dataset pascal_voc --net res101 \
#                    --checksession 1 --checkepoch 7 --checkpoint 10021 \
#                    --vis\
#                    --save_name origin\
#                    --cuda

python test_net.py --dataset pascal_voc --net res101 \
                   --checksession 1 --checkepoch 14 --checkpoint 10021 \
                   --flag bgul\
                   --save_name rec_bgul\
                   --vis\
                   --re_class\
                   --cuda

python test_net.py --dataset pascal_voc --net res101 \
                   --checksession 1 --checkepoch 7 --checkpoint 10021 \
                   --save_name origin\
                   --cuda


# python test_net.py --dataset pascal_voc --net res101 \
#                    --checksession 1 --checkepoch 8 --checkpoint 10021 \
#                    --cuda