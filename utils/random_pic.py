
import random


file_path = "/home/junjie/Code/faster-rcnn-gcn/data/VOCdevkit2007/VOC2007/ImageSets/Main/trainval.txt"
out_path = "/home/junjie/Code/faster-rcnn-gcn/data/VOCdevkit2007/VOC2007/ImageSets/Main/mintrain.txt"
file =  open(file_path,'r')
output = file.readlines()

tmp = random.sample(output, int(len(output) * 0.18))
out_file = open(out_path, 'w')
out_file.writelines(tmp)

out_file.close()
file.close()


