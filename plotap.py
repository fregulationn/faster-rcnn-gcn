import matplotlib.pyplot as plt
import pickle
import os
same_name_list = ['origin', 'rec_bgul']
out_put_dir = 'output/res101/voc_2007_test'
# class_name = [ 'aeroplane', 'bicycle', 'bird', 'boat',
#                          'bottle', 'bus', 'car', 'cat', 'chair',
#                          'cow', 'diningtable', 'dog', 'horse',
#                          'motorbike', 'person', 'pottedplant',
#                          'sheep', 'sofa', 'train', 'tvmonitor']
class_name = [ 'bottle', 'tvmonitor']


for i,name in enumerate(class_name):
    tmp_res = []
    # print("{}".format(name), end = "")
    tmp_res.append(name)
    for j,save_name in enumerate(same_name_list):
        pr = pickle.load(open(os.path.join(out_put_dir,save_name,name+'_pr.pkl'),'rb'))

    
        plt.plot(pr['rec'], pr['prec'], label = save_name + ":"+ str(pr['ap'] ))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend()
    # plt.title('Precision-Recall curve: AP={0:0.2f}'.format(pr['ap']))
    fn1 = os.path.join(out_put_dir, name + 'map.png')
    plt.savefig(fn1)
    plt.close(1)
