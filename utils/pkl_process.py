import pickle
import os
import numpy as np

same_name_list = ['lowshot1', 'lowshotreclass']
out_put_dir = 'output/vgg16/voc_2007_test'
class_name = [ 'aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor']

res = []
for same_name in same_name_list:
    print('\t%s'%same_name, end = "")
print("")

mAP  = np.zeros([len(same_name_list),20])

for i,name in enumerate(class_name):
    tmp_res = []
    # print("{}".format(name), end = "")
    tmp_res.append(name)
    for j,save_name in enumerate(same_name_list):
        pr = pickle.load(open(os.path.join(out_put_dir,save_name,name+'_pr.pkl'),'rb'))
        # print('\t%.3f'%pr['ap'], end = "")
        mAP[j][i] = pr['ap']
        tmp_res.append(round(pr['ap'],3))
    tmp_res.append(round(tmp_res[1] - tmp_res[2],3))
    res.append(tmp_res)
    print(tmp_res)
    # print(" ")



for k,same_name in enumerate(same_name_list):
    print('{:.4f}'.format(np.mean(mAP[k])), end = "")

# print(mAP)
# print(class_name)
# print(res)
