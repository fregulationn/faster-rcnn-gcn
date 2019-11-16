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
from model.rpn.bbox_transform import bbox_overlaps_batch, bbox_transform_batch, bbox_distances_batch
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
        # roi_part_match[0] and  roi_part_match_overlap[0] and so on

        if True:

            iou_threshold = 0.8
            dis_threshold = 0.2

            # first, calculate the overlaps among rois, set weights in edges between nodes iou>0.7 to 1
            overlaps = bbox_overlaps_batch(rois, rois)
            overlaps_bin = overlaps.cpu().data.numpy().copy()

            

            _, N_node, _ = overlaps.shape

            overlaps_bin1 = torch.unsqueeze(torch.eye(N_node, N_node).cuda(), dim = 0)
            overlaps_bin1[overlaps >= iou_threshold] = 1
            overlaps_bin1[overlaps < iou_threshold] = 0

            for j in range(N_node):
                for k in range(N_node):
                    if overlaps_bin[0][j][k] >= iou_threshold:
                        overlaps_bin[0][j][k] = 1
                    else:
                        overlaps_bin[0][j][k] = 0
                    if k == j:
                        overlaps_bin[0][j][k] = 0

            # second, calculate the distance among rois, set weights in edges between nodes iou=0 and dis<threshold to 1
            distances = bbox_distances_batch(rois, rois)
            distances_bin = distances.cpu().data.numpy().copy()

            for j in range(N_node):
                for k in range(N_node):
                    if distances_bin[0][j][k] <= dis_threshold:
                        distances_bin[0][j][k] = 1
                    else:
                        distances_bin[0][j][k] = 0
                    if k == j:
                        distances_bin[0][j][k] = 0

            #adj_matrix_bin = overlaps_bin + distances_bin

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

        dot_product_mat = torch.mm(pooled_feat, torch.transpose(pooled_feat, 0, 1))
        len_vec = torch.unsqueeze(torch.sqrt(torch.sum(pooled_feat * pooled_feat, dim=1)), dim=0)
        len_mat = torch.mm(torch.transpose(len_vec, 0, 1), len_vec)
        pooled_feat_sim_mat = dot_product_mat / len_mat

        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

        # update 20191027: build graph for rois based on index (default: batch_size = 1)
        part_size = 10
        relation_size = 5
        if True:
            cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

            # calculate the adj_mat based on adj_matrix_bin, the weights on edges are the cosine distance between nodes
            adj_matrix = np.zeros((N_node, N_node))

            for s in range(N_node):
                row_idx = [t for t in range(N_node)]
                random.shuffle(row_idx)
                part_cnt = 0
                relation_cnt = 0
                for t in row_idx:
                    if part_cnt <= part_size:
                        if overlaps_bin[0, s, t] == 1:
                            node_feat_s = pooled_feat[s, :]
                            node_feat_t = pooled_feat[t, :]
                            adj_matrix[s, t] = cos(node_feat_s, node_feat_t)
                            part_cnt = part_cnt + 1
                            continue
                for t in row_idx:
                    if part_cnt <= part_size:
                        if overlaps_bin[0, s, t] == 1:
                            node_feat_s = pooled_feat[s, :]
                            node_feat_t = pooled_feat[t, :]
                            adj_matrix[s, t] = cos(node_feat_s, node_feat_t)
                            part_cnt = part_cnt + 1
                            continue
                    # if relation_cnt <= relation_size:
                    #     if distances_bin[0, s, t] == 1:
                    #         node_feat_s = pooled_feat[s, :]
                    #         node_feat_t = pooled_feat[t, :]
                    #         adj_matrix[s, t] = cos(node_feat_s, node_feat_t)
                    #         relation_cnt = relation_cnt + 1
                    #         continue

                    # if part_cnt > part_size and relation_cnt > relation_size:
                    #     break
                    if part_cnt > part_size:
                        break

            adj_matrix = torch.from_numpy(adj_matrix).float().cuda()

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
