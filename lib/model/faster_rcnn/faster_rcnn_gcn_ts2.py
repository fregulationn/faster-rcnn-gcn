import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from model.rpn.bbox_transform import bbox_overlaps_batch, bbox_transform_batch
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta

def subgraph_split(adj):
    visited = np.zeros((adj.shape[0]))
    vertex_subgraph = np.zeros((adj.shape[0]))
    idx_subgraph = 0
    for i in range(adj.shape[0]):
        if visited[i] == 0:
            DFS(adj, i, visited, vertex_subgraph, idx_subgraph)
            idx_subgraph = idx_subgraph + 1

    return idx_subgraph, vertex_subgraph


def DFS(adj, i, visited, vertex_subgraph, idx_subgraph):
    visited[i] = 1
    vertex_subgraph[i] = idx_subgraph
    for j in range(adj.shape[0]):
        if visited[j] == 0 and adj[i][j] != 0:
            visited[j] = 1
            vertex_subgraph[j] = idx_subgraph
            DFS(adj, j, visited, vertex_subgraph, idx_subgraph)

class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)

        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)

        # update 20191026: get the index of nodes in graph for rois (default: batch_size = 1)
        # if we want to change batch_size, we should consider to change roi2gt_assignment[0]
        # roi_part_match[0] and  roi_part_match_overlap[0] and so onif self.training:

        part_threshold = 0.5

        # the shape of rois is 1,300,5, however, there is no 300 proposal after nms, so the last of the rois is all 0s
        rois_none_idx = 300
        for i in range(rois.shape[1]):
            if rois[:,i,:].sum() <= 0:
                rois_none_idx = i
                break

        # # first, calculate the overlaps among rois and gt, get the max roi for each gt (node_cls)
        overlaps = bbox_overlaps_batch(rois[:,:rois_none_idx,:], rois[:,:rois_none_idx,:])[0]

        N_node, _ = overlaps.shape

        overlaps_bin = overlaps.cpu().data.numpy().copy()

        for j in range(N_node):
            for k in range(N_node):
                if overlaps_bin[j][k] >= part_threshold:
                    overlaps_bin[j][k] = 1
                else:
                    overlaps_bin[j][k] = 0
                if k == j:
                    overlaps_bin[j][k] = 0

        idx_subgraph, vertex_subgraph = subgraph_split(overlaps_bin)

        # do roi pooling based on predicted rois
        if cfg.POOLING_MODE == 'crop':
            # pdb.set_trace()
            # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
            grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
            pooled_feat = self.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
        elif cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)

        # # update 20191105: build graph for rois based on index (default: batch_size = 1)

        roi_all_idx_list = []
        roi_cls_idx_list = []
        roi_part_idx_list = []

        adj_jud = np.zeros((0))
        adj_rois = torch.zeros(0).cuda().long()

        for k in range(idx_subgraph):
            idx_k = np.transpose(np.argwhere(vertex_subgraph == k))[0]
            roi_all_idx_list.append(idx_k)

        overlaps = overlaps.cpu().data.numpy()

        # 选度数最大的点作为node_cls
        for i in range(len(roi_all_idx_list)):
            rois_idx = roi_all_idx_list[i]

            # consider the size of rois_select larger than 5, the rois_select is probably an object
            if rois_idx.shape[0] < 5:
                continue

            overlaps_once = overlaps[rois_idx][:,rois_idx]
            overlaps_once_bin = overlaps_bin[rois_idx][:,rois_idx]

            N_node_once, _ = overlaps_once.shape

            ########## update 20191104: select IoU > threshold
            # for j in range(N_node_once):
            #     for k in range(N_node_once):
            #         if overlaps_once[j][k] >= part_threshold:
            #             overlaps_once[j][k] = 1
            #         else:
            #             overlaps_once[j][k] = 0
            #         if k == j:
            #             overlaps_once[j][k] = 0

            # overlaps_once = np.sum(overlaps_once, axis=1)
            #
            # rois_once_max_idx = np.argmax(overlaps_once)
            # roi_cls_idx_list.append(rois_idx[rois_once_max_idx])
            #
            # roi_part_tmp = []
            # for k in range(rois_idx.shape[0]):
            #     if overlaps[rois_idx[rois_once_max_idx]][k] == 0:
            #         continue
            #     roi_part_tmp.append(rois_idx[k])
            # roi_part_tmp = torch.from_numpy(np.array(roi_part_tmp))
            # roi_part_idx_list.append(roi_part_tmp)

            ########## update 20191107: all proposal

            overlaps_once_bin = np.sum(overlaps_once_bin, axis=1)

            rois_once_max_idx = np.argmax(overlaps_once_bin)
            roi_cls_idx_list.append(rois_idx[rois_once_max_idx])

            roi_part_tmp = []
            roi_iou = overlaps_once[rois_once_max_idx]
            roi_part_num_threshold = 10
            if roi_iou.shape[0] >= roi_part_num_threshold:
                roi_order = np.argsort(roi_iou)[::-1]
                for ii in range(roi_part_num_threshold):
                    roi_part_tmp.append(rois_idx[roi_order[ii]])
            else:
                for k in range(rois_idx.shape[0]):
                    if overlaps[rois_idx[rois_once_max_idx]][k] == 0:
                        continue
                    roi_part_tmp.append(rois_idx[k])
            roi_part_tmp = torch.from_numpy(np.array(roi_part_tmp))
            roi_part_idx_list.append(roi_part_tmp)

        roi_cls_idx_list = torch.from_numpy(np.array(roi_cls_idx_list)).cuda()

        for i in range(roi_cls_idx_list.shape[0]):
            adj_jud = np.concatenate((adj_jud, [1]))
            adj_rois = torch.cat((adj_rois, roi_cls_idx_list[i:i+1]))
            try:
                if roi_part_idx_list[i].shape[0] != 0:
                    adj_jud = np.concatenate((adj_jud, np.zeros((roi_part_idx_list[i].shape[0]))))
                    adj_rois = torch.cat((adj_rois, roi_part_idx_list[i].cuda()))
            except IndexError:
                print ('IndexError happen, continue')
                continue

        node_cls_idx = np.transpose(np.argwhere(adj_jud == 1))[0]

        adj_matrix_bin = np.zeros((len(adj_jud), len(adj_jud)))

        # link edges for node_cls to node_cls
        for k in range(len(node_cls_idx)-1):
            idx_node_cls_1 = node_cls_idx[k]
            idx_node_cls_2 = node_cls_idx[k + 1]
            adj_matrix_bin[idx_node_cls_1, idx_node_cls_2] = 1
            adj_matrix_bin[idx_node_cls_2, idx_node_cls_1] = 1

        # link edges for node_cls to related node_part
        for k in range(len(node_cls_idx)-1):
            idx_start = node_cls_idx[k]
            idx_end = node_cls_idx[k + 1]
            for s in range(idx_start, idx_end):
                for t in range(idx_start, idx_end):
                    if s == t:
                        adj_matrix_bin[s, t] = 0
                    else:
                        adj_matrix_bin[s, t] = 1

        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        adj_matrix = np.zeros((len(adj_jud), len(adj_jud)))

        for s in range(len(adj_jud)):
            for t in range(len(adj_jud)):
                if adj_matrix_bin[s, t] == 1:
                    node_feat_s = pooled_feat[adj_rois[s], :]
                    node_feat_t = pooled_feat[adj_rois[t], :]
                    adj_matrix[s, t] = cos(node_feat_s, node_feat_t)
                else:
                    adj_matrix[s, t] = 0

        adj_matrix = torch.from_numpy(adj_matrix).float().cuda()

        try:
            pooled_feat[adj_rois, :] = F.relu(self.gcn1(pooled_feat[adj_rois, :], adj_matrix))
            pooled_feat[adj_rois, :] = F.relu(self.gcn2(pooled_feat[adj_rois, :], adj_matrix))
        except RuntimeError:
            print (pooled_feat[adj_rois, :].size())
            print (adj_matrix.size())

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)


        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

            
        # update 2019-6-17:fix the bug for dimension specified as 0...
        if self.training:
            rpn_loss_cls = torch.unsqueeze(rpn_loss_cls, 0)
            rpn_loss_bbox = torch.unsqueeze(rpn_loss_bbox, 0)
            RCNN_loss_cls = torch.unsqueeze(RCNN_loss_cls, 0)
            RCNN_loss_bbox = torch.unsqueeze(RCNN_loss_bbox, 0)
            
        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
