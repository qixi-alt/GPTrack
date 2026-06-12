from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate
import os


class GPTrackActor(BaseActor):
    """ Actor for training GPTrack models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

        self.vis_mask = getattr(cfg, "VIS_MASK", False) if cfg is not None else False

        self.vis_dir = None
        if self.vis_mask:
            vis_dir_cfg = getattr(cfg, "VIS_MASK_DIR", None)
            if vis_dir_cfg is None:
                raise ValueError("VIS_MASK is True but VIS_MASK_DIR is not set in config")

            # Convert to an absolute path for safety.
            self.vis_dir = os.path.abspath(vis_dir_cfg)
            os.makedirs(self.vis_dir, exist_ok=True)

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data['visible'])

        return loss, status

    def forward_pass(self, data):
        # currently only support 1 template and 1 search region
        assert len(data['visible']['template_images']) == 1
        assert len(data['visible']['search_images']) == 1 or len(data['visible']['search_images']) == 2

        template_img_v = data['visible']['template_images'][0].view(-1, *data['visible']['template_images'].shape[2:])  # (batch, 3, 128, 128)
        template_img_i = data['infrared']['template_images'][0].view(-1, *data['infrared']['template_images'].shape[2:])  # (batch, 3, 128, 128)        
        
        search_img_v = data['visible']['search_images'][0].view(-1, *data['visible']['search_images'].shape[2:])  # (batch, 3, 320, 320)
        search_img_i = data['infrared']['search_images'][0].view(-1, *data['infrared']['search_images'].shape[2:])  # (batch, 3, 320, 320)
        search_img_v_last = data['visible']['search_images'][1].view(-1, *data['visible']['search_images'].shape[2:])  # (batch, 3, 320, 320)
        search_img_i_last = data['infrared']['search_images'][1].view(-1, *data['infrared']['search_images'].shape[2:])  # (batch, 3, 320, 320)

        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(self.cfg, template_img_v.shape[0], template_img_v.device,
                                            data['visible']['template_anno'][0])

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                                total_epochs=ce_start_epoch + ce_warm_epoch,
                                                ITERS_PER_EPOCH=1,
                                                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])

        out_dict = self.net(template=[template_img_v, template_img_i],
                            search=[search_img_v, search_img_i, search_img_v_last, search_img_i_last],
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=False)

        return out_dict

    def denorm_bbox_xywh(gt_bbox, H, W):
        """
        gt_bbox: (4,) normalized xywh
        """
        x, y, w, h = gt_bbox.tolist()
        return int(x * W), int(y * H), int(w * W), int(h * H)


    # def visualize_masks(self, masks, gt_bbox, epoch=0, max_show=4):
    #     import os
    #     import matplotlib.pyplot as plt
    #     import matplotlib.patches as patches
    #
    #     # Save each epoch in its own subdirectory.
    #     save_dir = os.path.join(self.vis_dir, f"epoch_{epoch:03d}")
    #     os.makedirs(save_dir, exist_ok=True)
    #
    #     gt = gt_bbox[0].detach().cpu()
    #     x, y, w, h = gt.tolist()
    #
    #     for i, m in enumerate(masks[:max_show]):
    #         if not isinstance(m, torch.Tensor):
    #             continue
    #
    #         mask = m[0, 0].detach().cpu().numpy()
    #         H, W = mask.shape
    #
    #         fig, ax = plt.subplots(1, figsize=(4, 4))
    #         ax.imshow(mask, cmap="jet")
    #         ax.axis("off")
    #
    #         rect = patches.Rectangle(
    #             (x * W, y * H),
    #             w * W,
    #             h * H,
    #             linewidth=2,
    #             edgecolor="lime",
    #             facecolor="none"
    #         )
    #         ax.add_patch(rect)
    #
    #         fname = os.path.join(save_dir, f"mask_{i}.png")
    #         plt.savefig(fname, bbox_inches="tight", dpi=150)
    #         plt.close(fig)



    def compute_losses(self, pred_dict, gt_dict, return_status=True):

        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][0]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap([gt_dict['search_anno'][0]], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)



        # weighted sum
        # loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss
        loss = (self.loss_weight['giou'] * giou_loss
                + self.loss_weight['l1'] * l1_loss
                + self.loss_weight['focal'] * location_loss)

        #----------------------------------------------------------
        # # === Get masks ===
        # if 'masks' in pred_dict:
        #     masks = pred_dict['masks']
        # elif 'extra' in pred_dict and isinstance(pred_dict['extra'], dict):
        #     masks = pred_dict['extra'].get('masks', [])
        # else:
        #     masks = []

        # === Get masks ===
        masks = pred_dict.get('masks', [])
        if not masks and 'extra' in pred_dict:
            masks = pred_dict['extra'].get('masks', [])



        # def compute_mask_loss(mask_list):
        #     if not mask_list:
        #         return torch.tensor(0.0, device=pred_boxes.device)
        #
        #     sparse_losses = []
        #     binary_losses = []
        #
        #     for mask_pair in mask_list:
        #         for k in ['mask_rgb', 'mask_tir']:
        #             if k in mask_pair:
        #                 m = mask_pair[k]
        #
        #                 # print("mask shape:", m.shape)
        #                 sparse_losses.append(torch.mean(torch.abs(m)))  # Sparsity.
        #                 binary_losses.append(torch.mean((m - 0.5) ** 2))  # Focus regularization.
        #
        #     sparse = torch.stack(sparse_losses).mean() if sparse_losses else torch.tensor(0.0, device=pred_boxes.device)
        #     # binary = - (m * torch.log(m + 1e-8) + (1 - m) * torch.log(1 - m + 1e-8)).mean()
        #
        #     binary = torch.stack(binary_losses).mean() if binary_losses else torch.tensor(0.0, device=pred_boxes.device)
        #     # print("current mask_loss =", sparse.item(), binary.item(), "total =", (sparse + binary).item())
        #
        #     return sparse + binary
        #     # return sparse

        def build_gt_mask(gt_bbox, Hm, Wm, device):
            """
            gt_bbox: (4,) normalized xywh
            Output: binary mask with shape [1, 1, Hm, Wm].
            """
            x, y, w, h = gt_bbox.tolist()
            gt_mask = torch.zeros((1, 1, Hm, Wm), device=device)

            x1 = int((x - w / 2) * Wm)
            y1 = int((y - h / 2) * Hm)
            x2 = int((x + w / 2) * Wm)
            y2 = int((y + h / 2) * Hm)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(Wm, x2), min(Hm, y2)

            gt_mask[:, :, y1:y2, x1:x2] = 1.0
            return gt_mask

        def compute_mask_loss(mask_list):
            if not mask_list:
                return torch.tensor(0.0, device=pred_boxes.device)

            eps = 1e-6
            losses = []

            for m in mask_list:
                if not isinstance(m, torch.Tensor):
                    continue

                # Sparsity.
                sparse = torch.mean(m)

                # Entropy term that encourages binarization.
                m_clamped = torch.clamp(m, eps, 1.0 - eps)
                binary = - (m_clamped * torch.log(m_clamped)
                            + (1 - m_clamped) * torch.log(1 - m_clamped)).mean()

                # Coverage term to prevent collapse to all zeros.
                coverage = torch.mean(m)
                penalty = (coverage - 0.15) ** 2

                # Assume m has shape [1, 1, Hm, Wm].
                gt_mask = build_gt_mask(gt_bbox[0], m.shape[-2], m.shape[-1], m.device)

                inside = (m * gt_mask).sum()
                total = m.sum() + 1e-6
                ratio = inside / total
                gt_guidance = -ratio

                losses.append(sparse + 0.01 * binary + penalty+ 0.05 * gt_guidance)

            if len(losses) == 0:
                return torch.tensor(0.0, device=pred_boxes.device)

            return torch.stack(losses).mean()

        # === Compute the auxiliary loss and add it to the total loss. ===
        mask_loss = compute_mask_loss(masks)

        # # ================== Visualization only; not used for training. ==================
        # if self.vis_mask:
        #     # Control frequency, for example every 20 iterations or epochs.
        #     epoch = gt_dict.get("epoch", None)
        #
        #
        #     if epoch is None or epoch % 20 == 0:
        #         self.visualize_masks(masks, gt_bbox)
        # # =========================================================

        loss = loss + self.loss_weight.get("mask", 1.0) * mask_loss
        #--------------------------------------------


        if return_status:
            # status for log
            # Helper for safely reading scalar values.
            def get_val(v):
                return v.item() if isinstance(v, torch.Tensor) else float(v)
            mean_iou = iou.detach().mean()

            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "IoU": mean_iou.item(),
                      "Loss/mask": get_val(mask_loss)}
            return loss, status
        else:
            return loss
