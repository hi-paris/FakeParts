# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Type

import torch
from torch import nn
from einops import rearrange
from models.lib.sam2.modeling.sam2_utils import LayerNorm2d, MLP


class MaskDecoder_custom(nn.Module):
    def __init__(
        self,
        *,
        # transformer_dim: int,
        transformer: nn.Module,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        # self.transformer_dim = transformer_dim
        self.transformer = transformer

        # self.num_multimask_outputs = num_multimask_outputs

        # self.iou_token = nn.Embedding(1, transformer_dim)
        # self.num_mask_tokens = num_multimask_outputs + 1
        # self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        # self.mask_tokens.weight.require_grad = False
        # self.iou_token.weight.require_grad = False
        
        
        # self.pred_obj_scores = pred_obj_scores
        # if self.pred_obj_scores:
        #     self.obj_score_token = nn.Embedding(1, transformer_dim)
            
        # self.mask_tokens.weight.require_grad = False
        # self.iou_token.weight.require_grad = False            
        # self.obj_score_token.weight.require_grad = False            
            
        # self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr

        # self.output_upscaling = nn.Sequential(
        #     nn.ConvTranspose2d(
        #         transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
        #     ),
        #     LayerNorm2d(transformer_dim // 4),
        #     activation(),
        #     nn.ConvTranspose2d(
        #         transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
        #     ),
        #     activation(),
        # )
        # self.use_high_res_features = use_high_res_features
        # if use_high_res_features:
        #     self.conv_s0 = nn.Conv2d(
        #         transformer_dim, transformer_dim // 8, kernel_size=1, stride=1
        #     )
        #     self.conv_s1 = nn.Conv2d(
        #         transformer_dim, transformer_dim // 4, kernel_size=1, stride=1
        #     )

        # self.output_hypernetworks_mlps = nn.ModuleList(
        #     [
        #         MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        #         for i in range(self.num_mask_tokens)
        #     ]
        # )

        # self.iou_prediction_head = MLP(
        #     transformer_dim,
        #     iou_head_hidden_dim,
        #     self.num_mask_tokens,
        #     iou_head_depth,
        #     sigmoid_output=iou_prediction_use_sigmoid,
        # )
        # if self.pred_obj_scores:
        #     self.pred_obj_score_head = nn.Linear(transformer_dim, 1)
        #     if pred_obj_scores_mlp:
        #         self.pred_obj_score_head = MLP(transformer_dim, transformer_dim, 1, 3)

        # # When outputting a single mask, optionally we can dynamically fall back to the best
        # # multimask output token if the single mask output token gives low stability scores.
        # self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        # self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
        # self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh

    # def forward(
    #     self,
    #     image_embeddings: torch.Tensor,
    #     image_pe: torch.Tensor,
    #     sparse_prompt_embeddings: torch.Tensor,
    #     dense_prompt_embeddings: torch.Tensor,
    #     multimask_output: bool,
    #     repeat_image: bool,
    #     high_res_features: Optional[List[torch.Tensor]] = None,
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Predict masks given image and prompt embeddings.

    #     Arguments:
    #       image_embeddings (torch.Tensor): the embeddings from the image encoder
    #       image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
    #       sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
    #       dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
    #       multimask_output (bool): Whether to return multiple masks or a single
    #         mask.

    #     Returns:
    #       torch.Tensor: batched predicted masks
    #       torch.Tensor: batched predictions of mask quality
    #       torch.Tensor: batched SAM token for mask output
    #     """
    #     masks, iou_pred, mask_tokens_out, object_score_logits, img_token_attn, attn_TokentoImage, attn_ImagetoToken = self.predict_masks(
    #         image_embeddings=image_embeddings,
    #         image_pe=image_pe,
    #         sparse_prompt_embeddings=sparse_prompt_embeddings,
    #         dense_prompt_embeddings=dense_prompt_embeddings,
    #         repeat_image=repeat_image,
    #         high_res_features=high_res_features,
    #     )

    #     # Select the correct mask or masks for output
    #     if multimask_output:
    #         masks = masks[:, 1:, :, :]
    #         iou_pred = iou_pred[:, 1:]
    #     elif self.dynamic_multimask_via_stability and not self.training:
    #         masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
    #     else:
    #         masks = masks[:, 0:1, :, :]
    #         iou_pred = iou_pred[:, 0:1]

    #     if multimask_output and self.use_multimask_token_for_obj_ptr:
    #         sam_tokens_out = mask_tokens_out[:, 1:]  # [b, 3, c] shape
    #     else:
    #         # Take the mask output token. Here we *always* use the token for single mask output.
    #         # At test time, even if we track after 1-click (and using multimask_output=True),
    #         # we still take the single mask token here. The rationale is that we always track
    #         # after multiple clicks during training, so the past tokens seen during training
    #         # are always the single mask token (and we'll let it be the object-memory token).
    #         sam_tokens_out = mask_tokens_out[:, 0:1]  # [b, 1, c] shape

    #     # Prepare output
    #     return masks, iou_pred, sam_tokens_out, object_score_logits, img_token_attn, attn_TokentoImage, attn_ImagetoToken

    def forward(
        self,
        query: torch.Tensor,
        # image_pe: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        # image_pe = image_pe.flatten(2).permute(0, 2, 1)
        # pos_src = torch.repeat_interleave(image_pe, query.shape[0], dim=0)
        # pos_src = torch.repeat_interleave(pos_src.unsqueeze(1), query.shape[1], dim=1)
        query, key, _, _ = self.transformer(key, query)
        return query, key




    def _get_stability_scores(self, mask_logits):
        """
        Compute stability scores of the mask logits based on the IoU between upper and
        lower thresholds, similar to https://github.com/fairinternal/onevision/pull/568.
        """
        mask_logits = mask_logits.flatten(-2)
        stability_delta = self.dynamic_multimask_stability_delta
        area_i = torch.sum(mask_logits > stability_delta, dim=-1).float()
        area_u = torch.sum(mask_logits > -stability_delta, dim=-1).float()
        stability_scores = torch.where(area_u > 0, area_i / area_u, 1.0)
        return stability_scores

    def _dynamic_multimask_via_stability(self, all_mask_logits, all_iou_scores):
        """
        When outputting a single mask, if the stability score from the current single-mask
        output (based on output token 0) falls below a threshold, we instead select from
        multi-mask outputs (based on output token 1~3) the mask with the highest predicted
        IoU score. This is intended to ensure a valid mask for both clicking and tracking.
        """
        # The best mask from multimask output tokens (1~3)
        multimask_logits = all_mask_logits[:, 1:, :, :]
        multimask_iou_scores = all_iou_scores[:, 1:]
        best_scores_inds = torch.argmax(multimask_iou_scores, dim=-1)
        batch_inds = torch.arange(
            multimask_iou_scores.size(0), device=all_iou_scores.device
        )
        best_multimask_logits = multimask_logits[batch_inds, best_scores_inds]
        best_multimask_logits = best_multimask_logits.unsqueeze(1)
        best_multimask_iou_scores = multimask_iou_scores[batch_inds, best_scores_inds]
        best_multimask_iou_scores = best_multimask_iou_scores.unsqueeze(1)

        # The mask from singlemask output token 0 and its stability score
        singlemask_logits = all_mask_logits[:, 0:1, :, :]
        singlemask_iou_scores = all_iou_scores[:, 0:1]
        stability_scores = self._get_stability_scores(singlemask_logits)
        is_stable = stability_scores >= self.dynamic_multimask_stability_thresh

        # Dynamically fall back to best multimask output upon low stability scores.
        mask_logits_out = torch.where(
            is_stable[..., None, None].expand_as(singlemask_logits),
            singlemask_logits,
            best_multimask_logits,
        )
        iou_scores_out = torch.where(
            is_stable.expand_as(singlemask_iou_scores),
            singlemask_iou_scores,
            best_multimask_iou_scores,
        )
        return mask_logits_out, iou_scores_out
