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
from model.faster_rcnn.cgcn_models import CGCN
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pickle
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

        
        if cfg.GCN.RE_CLASS:
            self.Class_GCN = CGCN(cfg.GCN.N_FEAT, cfg.GCN.N_HID, cfg.GCN.DROPOUT, self.n_classes, t = 0.05, adj_file = cfg.GCN.ADJ_FILE)

    # def forward(self, im_data, im_info, gt_boxes, num_boxes, gt_classes):
    # def forward(self, im_data, im_info, gt_boxes, num_boxes, iteration, mlgcn_threshold):
    # def forward(self, im_data, im_info, gt_boxes, num_boxes, mlgcn_threshold):
    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)

        # feed base feature map to RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

        # if it is training phase, then use ground truth bboxes for refining
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

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)

        # cls_re_score, regular_term = self.Class_GCN(cls_score[:,1:])
        # new_cls_score = torch.cat((cls_score[:,0].view(-1,1),cls_re_score),dim = -1)
        # cls_prob = F.softmax(new_cls_score, 1)  

        cls_prob = F.softmax(cls_score, 1)  
        
        # # cls_prob_old = F.softmax(cls_score, dim=1)
        # if cfg.GCN.RE_CLASS:
        #     cls_reclass_score, regular_term = self.Class_GCN(cls_prob_old)
        
        # max min
        # cls_max = torch.max(cls_score, dim = 1)[0].view(-1, 1)
        # cls_min = torch.min(cls_score, dim = 1)[0].view(-1, 1)
        # cls_prob = (cls_score - cls_min)/(cls_max - cls_min)
        
        # # div sum 
        # cls_reclass_sum = torch.sum(cls_reclass_score, dim = 1 ).view(-1, 1)
        # cls_reclass_prob = torch.div(cls_reclass_score, cls_reclass_sum)
        
        # ## ml-gcn
        # cls_feature = pickle.load(open('data/VOCdevkit2007/VOC2007/feature.pkl', 'rb'))
        # cls_feature = torch.from_numpy(cls_feature).cuda()
        # cls_gcn_scores = torch.mm(pooled_feat, cls_feature)
        # # cls_gcn_prob = F.softmax(cls_gcn_scores, 1)
        # cls_gcn_max = torch.max(cls_gcn_scores, dim = 1)[0].view(-1, 1)
        # cls_gcn_min = torch.min(cls_gcn_scores, dim = 1)[0].view(-1, 1)
        # cls_gcn_prob = (cls_gcn_scores - cls_gcn_min)/(cls_gcn_max - cls_gcn_min)
        
        
        # # ml-gcn
        # _, max_index = torch.max(cls_prob, dim = 1)
        # col_mask = torch.ne(max_index, 0)
        # mask = col_mask.view(-1, 1).repeat(1, 21)
        # cls_prob = torch.where(mask, cls_prob, cls_gcn_prob)

        # ## reclass select bg
        # _, max_index = torch.max(cls_reclass_score, dim = 1)
        # col_mask = torch.ne(max_index, 0)
        # mask = col_mask.view(-1, 1).repeat(1, 21)
        # cls_score = torch.where(mask, cls_reclass_score, cls_prob_old)
        # cls_prob = torch.where(mask, cls_reclass_prob, cls_prob_old)


        # # gt assign
        # gt_assign = torch.zeros(self.n_classes).cuda().float()
        # gt_assign[gt_classes] = 1
        # gt_reclass_prob = torch.mul(cls_prob_old, gt_assign)
        # # gt_reclass_socre = torch.sum(gt_reclass_socre, dim = 1 ).view(-1, 1)
        # # gt_reclass_prob = torch.div(gt_reclass_socre, gt_reclass_sum)
        
        # # ml-gcn assign
        # mlgcn_assign = pickle.load(open("/home/junjie/Framework/ML_GCN/data/{}.pkl".format(iteration), 'rb'))
        # mlgcn_assign = F.softmax(torch.from_numpy(mlgcn_assign), dim = 1)
        # mlgcn_assign = torch.cat((torch.tensor(0.0).view(1, -1), mlgcn_assign), dim = -1).float().cuda()
        # mlgcn_assign[ mlgcn_assign >= mlgcn_threshold] = 1
        # mlgcn_assign[ mlgcn_assign < mlgcn_threshold] = 0
        # gt_reclass_prob = torch.mul(cls_reclass_prob, mlgcn_assign)
        # # gt_reclass_socre = torch.sum(gt_reclass_socre, dim = 1 ).view(-1, 1)
        # # gt_reclass_prob = torch.div(gt_reclass_socre, gt_reclass_sum)
        # max_index = torch.argmax(cls_prob_old, dim = 1)
        # col_mask = torch.ne(max_index, 0)
        # mask = col_mask.view(-1, 1).repeat(1, 21)
        # cls_prob = torch.where(mask, gt_reclass_prob, cls_prob_old)

        # ml-gcn-before head
        # cls_feature = pickle.load(open('data/VOCdevkit2007/VOC2007/feature.pkl', 'rb'))
        # cls_feature = torch.from_numpy(cls_feature).cuda()
        # cls_pool = nn.AdaptiveMaxPool2d((2, 1))
        # base_feat = cls_pool(base_feat).view(base_feat.size(0), -1)
        # cls_gcn_scores = torch.mm(base_feat, cls_feature)
        # cls_gcn_prob = F.softmax(cls_gcn_scores, 1)

        # # mlgcn_threshold = 0.05
        # cls_gcn_prob[ cls_gcn_prob >= mlgcn_threshold] = 1
        # cls_gcn_prob[ cls_gcn_prob < mlgcn_threshold] = 0
        # cls_gcn_prob = torch.cat((torch.ones(300, 1).float().cuda(),cls_gcn_prob),dim = -1)
        # gcn_reclass_prob = torch.mul(cls_gcn_prob, cls_prob_old)
        # max_index = torch.argmax(cls_prob_old, dim = 1)
        # col_mask = torch.ne(max_index, 0)
        # mask = col_mask.view(-1, 1).repeat(1, 21)
        # cls_prob = torch.where(mask, gcn_reclass_prob, cls_prob_old)


        # cls_gcn_max = torch.max(cls_gcn_scores, dim = 1)[0].view(-1, 1)
        # cls_gcn_min = torch.min(cls_gcn_scores, dim = 1)[0].view(-1, 1)
        # cls_gcn_prob = (cls_gcn_scores - cls_gcn_min)/(cls_gcn_max - cls_gcn_min)
        # cls_gcn_prob = torch.cat((torch.zeros(300, 1).float().cuda(), cls_gcn_prob), dim = 1)
     
        
        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0
        RCNN_loss_regular = 0

        if self.training:
            # classification loss
            if cfg.GCN.RE_CLASS:
                RCNN_loss_regular = regular_term
            
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
            
            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

            rpn_loss_cls = torch.unsqueeze(rpn_loss_cls, 0)
            rpn_loss_bbox = torch.unsqueeze(rpn_loss_bbox, 0)
            RCNN_loss_cls = torch.unsqueeze(RCNN_loss_cls, 0)
            RCNN_loss_bbox = torch.unsqueeze(RCNN_loss_bbox, 0)

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, RCNN_loss_regular, rois_label

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
