

import argparse
import _init_paths
import numpy as np
import pickle
from model.utils.config import cfg
from roi_data_layer.roidb import combined_roidb

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate correlation matrix')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_trainval"
    
    # word2vec = pickle.load(open("/home/junjie/Framework/ML_GCN/data/voc/voc_glove_word2vec.pkl", 'rb'))
    mlgcn_matrix = pickle.load(open("/home/junjie/Framework/ML_GCN/data/coco/coco_adj.pkl", 'rb'))
    # with open("rb") as f:
    #      = pickle.loads(f,pickle.HIGHEST_PROTOCOL)


    tao = 0.4
    cfg.TRAIN.USE_FLIPPED = False
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name, False)
    imdb.competition_mode(on=True)
    num_images = len(imdb.image_index)
    print('{:d} roidb entries'.format(len(roidb)))
    
    cor_matrix = np.zeros([imdb.num_classes, imdb.num_classes])
    M_matrx = np.zeros([imdb.num_classes, imdb.num_classes]).astype(int)
    N_list = np.zeros([imdb.num_classes]).astype(int)

    # for i in range(num_images):
    #     image_gt_classes = imdb.roidb[i]['gt_classes']
    #     image_gt_classes = list(set(image_gt_classes))

    #     for j in range(len(image_gt_classes)):
    #         index_j = image_gt_classes[j]
    #         N_list[index_j] += 1
    #         # M_matrx[index_j, index_j] += 1
    #         for k in range(j + 1, len(image_gt_classes)):
    #             index_k = image_gt_classes[k]
    #             M_matrx[index_j, index_k] += 1
    #             M_matrx[index_k, index_j] += 1
    
    # M_matrx[1:, 1:] = mlgcn_matrix['adj']
    # N_list[1:] =mlgcn_matrix['nums']
    # N_list[0] = num_images
    

    # with open('data/VOCdevkit2007/VOC2007/voc_adj_bg.pkl','wb') as f:
    #     pickle.dump({'adj':M_matrx, 'nums':N_list}, f)
    
    # M_matrx[0, 1:] = N_list[1:]
    # with open('data/VOCdevkit2007/VOC2007/voc_adj_bg_up.pkl','wb') as f:
    #     pickle.dump({'adj':M_matrx, 'nums':N_list}, f)

    coco_index = [1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72]
    coco_index = np.array(coco_index) -1

    M_matrx[1:, 1:] = mlgcn_matrix['adj'][coco_index-1, :][:, coco_index-1]
    N_list[1:] =mlgcn_matrix['nums'][coco_index-1]
    N_list[0] = 1

    # M_matrx[1:, 0] = N_list[1:]
    with open('data/VOCdevkit2007/VOC2007/voc_adj_bg_coco_0.pkl','wb') as f:
        pickle.dump({'adj':M_matrx, 'nums':N_list}, f)


    t = 0.05
    _adj = M_matrx
    _nums = N_list
    _nums = _nums[:, np.newaxis]
    _adj = _adj / _nums
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1
    _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
    _adj[1:, 1:] = _adj[1:, 1:] + np.identity(20, np.int)
    # cor_matrix = M_matrx / np.tile(N_list.reshape(-1,1),(1,imdb.num_classes))
    # sub_cor_matrix = cor_matrix[1:, 1:]
    # binary_matrix = np.where(sub_cor_matrix >= tao, 1, 0)

    # D = np.pow(A.sum(1).float(), -0.5)
    # D = np.diag(D)
    # adj = np.matmul(np.matmul(A, D).t(), D)
        
   
    print('end')
