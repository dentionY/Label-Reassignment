import torch
import torch.nn.functional as F

from ...loss.iou_loss import bbox_overlaps
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


class DynamicSoftLabelAssigner(BaseAssigner):
    """Computes matching between predictions and ground truth with
    dynamic soft label assignment.

    Args:
        topk (int): Select top-k predictions to calculate dynamic k
            best matchs for each gt. Default 13.
        iou_factor (float): The scale factor of iou cost. Default 3.0.
    """

    def __init__(self, topk=13, iou_factor=3.0, ignore_iof_thr=-1):
        self.topk = topk
        self.iou_factor = iou_factor
        self.ignore_iof_thr = ignore_iof_thr

    def assign(
        self,
        pred_scores,
        priors,
        decoded_bboxes,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
    ):
        """Assign gt to priors with dynamic soft label assignment.
        Args:
            pred_scores (Tensor): Classification scores of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            priors (Tensor): All priors of one image, a 2D-Tensor with shape
                [num_priors, 4] in [cx, cy, stride_w, stride_y] format.
            decoded_bboxes (Tensor): Predicted bboxes, a 2D-Tensor with shape
                [num_priors, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
                with shape [num_gts, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth labels of one image, a Tensor
                with shape [num_gts].

        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        INF = 100000000
        num_gt = gt_bboxes.size(0)
        num_bboxes = decoded_bboxes.size(0)


        assigned_gt_inds = decoded_bboxes.new_full((num_bboxes,), 0, dtype=torch.long)  

        prior_center = priors[:, :2]
        lt_ = prior_center[:, None] - gt_bboxes[:, :2]    
        rb_ = gt_bboxes[:, 2:] - prior_center[:, None]   

        deltas = torch.cat([lt_, rb_], dim=-1)
        is_in_gts = deltas.min(dim=-1).values > 0
        valid_mask = is_in_gts.sum(dim=1) > 0

        non_valid_mask = ~valid_mask
        non_valid_decoded_bbox = decoded_bboxes[non_valid_mask]

        valid_decoded_bbox = decoded_bboxes[valid_mask]                   
        valid_pred_scores = pred_scores[valid_mask]                      
        num_valid = valid_decoded_bbox.size(0)


        if num_gt == 0 or num_bboxes == 0 or num_valid == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes,))
            if num_gt == 0:
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = decoded_bboxes.new_full(
                    (num_bboxes,), -1, dtype=torch.long
                )
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels
            )

        pairwise_ious = bbox_overlaps(valid_decoded_bbox, gt_bboxes)           
        iou_cost = -torch.log(pairwise_ious + 1e-7)

        non_pairwise_ious = bbox_overlaps(non_valid_decoded_bbox, gt_bboxes)
        gt_onehot_label = (
            F.one_hot(gt_labels.to(torch.int64), pred_scores.shape[-1])
            .float()
            .unsqueeze(0)
            .repeat(num_valid, 1, 1)
        )                                                                           # [num_valid, num_gt, num_cls]
        valid_pred_scores = valid_pred_scores.unsqueeze(1).repeat(1, num_gt, 1)     # [num_valid, num_gt, num_cls]

        soft_label = gt_onehot_label * pairwise_ious[..., None]
        scale_factor = soft_label - valid_pred_scores.sigmoid()

        cls_cost = F.binary_cross_entropy_with_logits(
            valid_pred_scores, soft_label, reduction="none"
        ) * scale_factor.abs().pow(2.0)

        cls_cost = cls_cost.sum(dim=-1)

        cost_matrix = cls_cost + iou_cost * self.iou_factor                             # [num_valid, num_gt]
        matched_pred_ious, matched_gt_inds, non_matched_gt_inds, non_matched_pred_ious = self.dynamic_k_matching(
            cost_matrix, pairwise_ious, num_gt, valid_mask, non_pairwise_ious, non_valid_mask
        )
        
        #print("matched_pred_ious is ", matched_pred_ious)
        #print("non_matched_pred_ious is ", non_matched_pred_ious)
        #print("matched_gt_inds is ", matched_gt_inds)
        #print("non_matched_gt_inds is ", non_matched_gt_inds)

        assigned_gt_inds[valid_mask] = matched_gt_inds + 1                 
        assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
        assigned_labels[valid_mask] = gt_labels[matched_gt_inds].long()  
        max_overlaps = assigned_gt_inds.new_full(
            (num_bboxes,), -INF, dtype=torch.float32
        )
        max_overlaps[valid_mask] = matched_pred_ious
        assigned_gt_inds[non_valid_mask] = non_matched_gt_inds + 1
        assigned_labels[non_valid_mask] = gt_labels[non_matched_gt_inds].long()  
        max_overlaps[non_valid_mask] = non_matched_pred_ious  
        if (
            self.ignore_iof_thr > 0
            and gt_bboxes_ignore is not None
            and gt_bboxes_ignore.numel() > 0
            and num_bboxes > 0
        ):
            ignore_overlaps = bbox_overlaps(
                valid_decoded_bbox, gt_bboxes_ignore, mode="iof"
            )
            ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            ignore_idxs = ignore_max_overlaps > self.ignore_iof_thr
            assigned_gt_inds[ignore_idxs] = -1

            non_ignore_overlaps = bbox_overlaps(
                non_valid_decoded_bbox, gt_bboxes_ignore, mode="iof"
            )
            non_ignore_max_overlaps, _ = non_ignore_overlaps.max(dim=1)
            non_ignore_idxs = non_ignore_max_overlaps > self.ignore_iof_thr
            assigned_gt_inds[non_ignore_idxs] = -1 
        return AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels
        )
   def dynamic_k_matching(self, cost, pairwise_ious, num_gt, valid_mask, non_pairwise_ious, non_valid_mask):
        """Use sum of topk pred iou as dynamic k. Refer from OTA and YOLOX.

        Args:
            cost (Tensor): Cost matrix.
            pairwise_ious (Tensor): Pairwise iou matrix.
            num_gt (int): Number of gt.
            valid_mask (Tensor): Mask for valid bboxes.
        """
        matching_matrix = torch.zeros_like(cost)

        extra_matrix = torch.zeros_like(non_pairwise_ious)
        candidate_topk = min(self.topk, pairwise_ious.size(0))
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=0)
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
            )
            matching_matrix[:, gt_idx][pos_idx] = 1.0 

        del topk_ious, dynamic_ks, pos_idx

        prior_match_gt_mask = matching_matrix.sum(1) > 1 
        if prior_match_gt_mask.sum() > 0:
            num_valid = prior_match_gt_mask.size(0)
            exception_count = 0
            for i in range(num_valid):
                if prior_match_gt_mask[i] == True:
                    exception_count += 1
                    #print("i is ", i)
                    origin_matching_gt = torch.nonzero(matching_matrix[i]==1).squeeze()  
                    cost_min, replace_matching_gt = torch.min(cost[i, :], dim=0)
                    if replace_matching_gt in origin_matching_gt: 
                        #print("replace_matching_gt is ", replace_matching_gt)
                        #print("origin_matching_gt is ", origin_matching_gt)
                        index_replace_matching_gt = torch.nonzero(origin_matching_gt == replace_matching_gt).squeeze()
                        part_one_origin_matching_gt = origin_matching_gt[0:index_replace_matching_gt]
                        part_two_origin_matching_gt = origin_matching_gt[index_replace_matching_gt+1:]
                        new_origin_matching_gt = torch.cat((part_one_origin_matching_gt, part_two_origin_matching_gt),dim=0)
                        #print("new_origin_matching_gt is ", new_origin_matching_gt)

                        replace_matching_gt_num_prior = int(torch.sum(matching_matrix[:, replace_matching_gt]))   
                        new_origin_matching_gt_num_prior = torch.sum(matching_matrix.T[new_origin_matching_gt]).long()   
                        #print("torch.max(new_origin_matching_gt_num_prior) is ", torch.max(new_origin_matching_gt_num_prior))
                        #print("replace_matching_gt_num_prior - new_origin_matching_gt_num_prior is ", replace_matching_gt_num_prior - new_origin_matching_gt_num_prior)
                        ## Step 1. 
                        if replace_matching_gt_num_prior - torch.max(new_origin_matching_gt_num_prior) > 0: 
                            #print("replace_matching_gt_num_prior is ", replace_matching_gt_num_prior)
                            #print("new_origin_matching_gt_num_prior is ", new_origin_matching_gt_num_prior)
                            matching_matrix[i, :] *= 0.0
                            #matching_matrix[i, replace_matching_gt] = 0.0         
                            #print("new_origin_matching_gt is ", new_origin_matching_gt)
                            new_permuted_index = torch.randperm(len(new_origin_matching_gt))
                            new_index_chosen = new_permuted_index[0]
                            matching_matrix[i, new_origin_matching_gt[new_index_chosen]] = 1.0    
                            #print("new_index_chosen is ", new_index_chosen)
                            if len(new_origin_matching_gt) > 1:   
                                index_delete = (new_origin_matching_gt == new_origin_matching_gt[new_index_chosen]).nonzero().squeeze()
                                part_one_new_matching_gt = new_origin_matching_gt[0:index_delete]
                                part_two_new_matching_gt = new_origin_matching_gt[index_delete+1:]
                                new_origin_matching_gt = torch.cat((part_one_new_matching_gt, part_two_new_matching_gt), dim=0)  
                                distrib_signal = 1
                            else:
                                distrib_signal = 0
                        elif replace_matching_gt_num_prior - torch.max(new_origin_matching_gt_num_prior) < 0: 
                            matching_matrix[i, :] *= 0.0
                            matching_matrix[i, replace_matching_gt] = 1.0         
                            distrib_signal = 1
                        else:
                            matching_matrix[i, :] *= 0.0
                            permuted_list = torch.randperm(len(origin_matching_gt))   
                            new_index_chosen = permuted_list[0]
                            matching_matrix[i, origin_matching_gt[new_index_chosen]] = 1.0           
                            index_delete = (origin_matching_gt == origin_matching_gt[new_index_chosen]).nonzero().squeeze()
                            part_one_new_matching_gt = origin_matching_gt[0:index_delete]
                            part_two_new_matching_gt = origin_matching_gt[index_delete+1:]
                            new_origin_matching_gt = torch.cat((part_one_new_matching_gt, part_two_new_matching_gt), dim=0) 
                            distrib_signal = 1

                        ## Step 2. 
                        if distrib_signal == 1:
                            for index_gt in new_origin_matching_gt:   
                                non_iou_value, index_non_prior = torch.topk(non_pairwise_ious[:, index_gt], 3)   
                                non_zero_iou_value_index = torch.nonzero(non_iou_value>0.3).squeeze()
                                if non_zero_iou_value_index.numel() > 0:
                                    extra_matrix[index_non_prior[non_zero_iou_value_index], index_gt] = 1.0  
                                else:
                                    extra_matrix[index_non_prior, index_gt] = 0.0
                        else:
                            pass
                    else:                                      
                        permuted_index = torch.randperm(len(origin_matching_gt))
                        index_chosen = permuted_index[0]
                        matching_matrix[i, :] *= 0.0
                        matching_matrix[i, origin_matching_gt[index_chosen]] = 1.0
                        index_replace_matching_gt = torch.nonzero(origin_matching_gt == origin_matching_gt[0]).squeeze()
                        part_one_origin_matching_gt = origin_matching_gt[0:index_replace_matching_gt]
                        part_two_origin_matching_gt = origin_matching_gt[index_replace_matching_gt+1:]
                        new_origin_matching_gt = torch.cat((part_one_origin_matching_gt, part_two_origin_matching_gt),dim=0)
                        for index_gt in new_origin_matching_gt:   
                            non_iou_value, index_non_prior = torch.topk(non_pairwise_ious[:, index_gt], 3)   
                            non_zero_iou_value_index = torch.nonzero(non_iou_value>0.3).squeeze()
                            if non_zero_iou_value_index.numel() > 0:
                                extra_matrix[index_non_prior[non_zero_iou_value_index], index_gt] = 1.0 
                            else:
                                extra_matrix[index_non_prior, index_gt] = 0.0

        
        extra_prior_match_gt_mask = extra_matrix.sum(1) > 1   
        extra_prior_match_gt_mask = extra_prior_match_gt_mask.long()
        index_extra_prior_match_gt_mask = torch.nonzero(extra_prior_match_gt_mask == 1).squeeze()
        if extra_prior_match_gt_mask.sum() > 0:
            #print("index_extra_prior_match_gt_mask is ", index_extra_prior_match_gt_mask)
            if index_extra_prior_match_gt_mask.numel() == 1:
                index_extra_prior_match_gt_mask = index_extra_prior_match_gt_mask.unsqueeze(0)
            for index_nonzero in index_extra_prior_match_gt_mask:
                index_real_gt = torch.nonzero(extra_matrix[index_nonzero, :] == 1).squeeze()
                permuted_indices = torch.randperm(len(index_real_gt))
                index_random = permuted_indices[0]
                extra_matrix[index_nonzero, index_real_gt[index_random]] = 1.0     
                extra_matrix[index_nonzero, index_real_gt[0: index_random]] = 0.0    
                extra_matrix[index_nonzero, index_real_gt[index_random+1 : ]] = 0.0

        fg_mask_inboxes = matching_matrix.sum(1) > 0.0            
        valid_mask[valid_mask.clone()] = fg_mask_inboxes              
        matched_gt_inds = matching_matrix[fg_mask_inboxes, :].argmax(1)
        matched_pred_ious = (matching_matrix * pairwise_ious).sum(1)[fg_mask_inboxes]
        fg_non_valid_mask_inboxes = extra_matrix.sum(1) > 0.0
        non_valid_mask[non_valid_mask.clone()] = fg_non_valid_mask_inboxes
        non_matched_gt_inds = extra_matrix[fg_non_valid_mask_inboxes, :].argmax(1)
        non_matched_pred_ious = (extra_matrix * non_pairwise_ious).sum(1)[fg_non_valid_mask_inboxes]


        return matched_pred_ious, matched_gt_inds, non_matched_gt_inds, non_matched_pred_ious
