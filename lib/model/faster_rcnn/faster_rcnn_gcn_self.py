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
from model.rpn.bbox_distance import bbox_distances_batch
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta

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
        rois, rpn_loss_cls, rpn_loss_bbox, num_proposal = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

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

        # update 20191026: get the index of nodes in graph for rois (default: batch_size = 1)
        # if we want to change batch_size, we should consider to change roi2gt_assignment[0]
        # roi_part_match[0] and  roi_part_match_overlap[0] and so on

        iou_threshold = 0.7
        dis_threshold = 0.01
        # part_size = 10
        # relation_size = 5
        iou_size = 6
        edge_size = 4
        child_size = 4
        batch = 0
        if True:
            if not self.training:
                rois = rois[:, :num_proposal, :]
                pooled_feat = pooled_feat[:num_proposal,:]
    
            # first, calculate the overlaps among rois, set weights in edges between nodes iou>0.7 to 1
            overlaps = bbox_overlaps_batch(rois, rois)
            # overlaps_bin = overlaps.cpu().data.numpy().copy()
            
            _, N_node, _ = overlaps.shape
            # second, calculate the distance among rois, set weights in edges between nodes iou=0 and 
            distances = bbox_distances_batch(rois, rois)
            # update 20191115: build graph for rois based on index (default: batch_size = 1)
            # feature cosine similarity

            # similarity in PGCN
            dot_product_mat = torch.mm(pooled_feat, torch.transpose(pooled_feat, 0, 1))
            len_vec = torch.unsqueeze(torch.sqrt(torch.sum(pooled_feat * pooled_feat, dim=1)), dim=0)
            len_mat = torch.mm(torch.transpose(len_vec, 0, 1), len_vec)
            pooled_feat_sim_mat = dot_product_mat / len_mat

            # cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

            # calculate the adj_mat based on iou and distance, the weights on edges are the cosine similarity between nodes
            mask = torch.eye(N_node, N_node).cuda()
            for s in range(N_node):

                overlap_node_index = (overlaps[batch][s] >= iou_threshold).nonzero()
                overlap_node_size = iou_size if overlap_node_index.shape[0] > iou_size else overlap_node_index.shape[0]
                overlap_node_random = torch.randperm(overlap_node_index.shape[0])[0:overlap_node_size]
                overlap_node_index_select = overlap_node_index[overlap_node_random]

                # TODO(junjie) remove the iou box in distance box.

                distance_node_index = (distances[batch][s] < dis_threshold).nonzero()
                distance_node_size = iou_size if distance_node_index.shape[0] > iou_size else distance_node_index.shape[0]
                distance_node_random = torch.randperm(distance_node_index.shape[0])[0:distance_node_size]
                distance_node_index_select = distance_node_index[distance_node_random]


                _node_index_select = torch.cat((overlap_node_index_select, distance_node_index_select), dim = 0)
                if _node_index_select.shape[0] == 0:
                    continue
                else:
                    _node_index_select = _node_index_select.squeeze(dim = 1)
                _node_size = child_size if _node_index_select.shape[0] > child_size else _node_index_select.shape[0]
                _node_index_select_random = torch.randperm(_node_index_select.shape[0])[0:_node_size]
                node_index_select = _node_index_select[_node_index_select_random]

                mask[s,node_index_select] = 1
                # print("test ")

            adj_matrix = torch.mul(mask, pooled_feat_sim_mat)

            pooled_feat = F.relu(self.gcn1(pooled_feat, adj_matrix))
            pooled_feat = F.relu(self.gcn2(pooled_feat, adj_matrix))

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
