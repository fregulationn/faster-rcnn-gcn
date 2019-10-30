# python test_net.py --dataset pascal_voc --net res101 \
#                    --checksession 1 --checkepoch 7 --checkpoint 10021 \
#                    --vis\
#                    --save_name origin\
#                    --cuda

# python test_net.py --dataset pascal_voc --net res101 \
#                    --checksession 1 --checkepoch 14 --checkpoint 10021 \
#                    --flag bgul\
#                    --save_name rec_bgul\
#                    --re_class\
#                    --cuda

UDA_VISIBLE_DEVICES=0 python test_net.py --dataset pascal_voc --net res101 \
                    --checksession 1 --checkepoch 7 --checkpoint 10021 \
                    --flag t05\
                    --cuda

# python test_net.py --dataset pascal_voc --net res101 \
#                    --checksession 1 --checkepoch 8 --checkpoint 10021 \
#                    --cuda