import torch
import torch.nn as nn
from models.util import *
import torch.nn.functional as F
from models.lib.beit2 import modeling_finetune
from models.lib.sam2.modeling.sam.transformer import RoPEAttention
from models.lib.sam2.modeling.sam2_utils import get_activation_fn, get_clones
from typing import Optional
import torch
from torch import nn, Tensor
from models.lib.sam2.modeling import position_encoding
from einops import rearrange
import copy
import timm


class MemoryAttentionLayer_custom(nn.Module):
    def __init__(
        self,
        d_model: 256,
        dim_feedforward: 2048,
        dropout: 0.1,
        pos_enc_at_attn: False,
        pos_enc_at_cross_attn_keys: True,
        pos_enc_at_cross_attn_queries: False,
    ):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        self.self_attn = RoPEAttention(rope_theta=10000.0, feat_sizes=[16,16], num_heads=1,  downsample_rate=1, dropout=0.1, embedding_dim=768)
        self.cross_attn_image = RoPEAttention(rope_theta=10000.0, feat_sizes=[16,16], rope_k_repeat=True, num_heads=1, kv_in_dim=768, downsample_rate=1, dropout=0.1, embedding_dim=768)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation_str = "relu"
        self.activation = get_activation_fn(self.activation_str)
        # Where to add pos enc
        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.mask_token.requires_grad = False
    def _forward_sa(self, tgt, query_pos):
        # Self-Attention
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2,attn = self.self_attn(q, k, v=tgt2)
        tgt = tgt + self.dropout1(tgt2)
        return tgt,attn

    def _forward_ca(self, tgt, memory, query_pos, pos, mask=None, maskID=None, num_k_exclude_rope=0, clip_T=8, bs=8):
        kwds = {}
        if num_k_exclude_rope > 0:
            assert isinstance(self.cross_attn_image, RoPEAttention)
            kwds = {"num_k_exclude_rope": num_k_exclude_rope}
        if mask is not None:
            start_id = 0
            cat_tensor = []
            for i in range(bs):
                x_ = torch.cat([tgt[start_id:start_id+maskID[i]], self.mask_token.repeat(8-maskID[i],197, 1)], dim=0)
                start_id += maskID[i]
                cat_tensor.append(x_)
            tgt = torch.stack(cat_tensor)

            start_id = 0
            cat_tensor = []
            for i in range(bs): # b = 16
                x_ = torch.cat([memory[start_id:start_id+maskID[i]], self.mask_token.repeat(8-maskID[i],197, 1)], dim=0)
                start_id += maskID[i]
                cat_tensor.append(x_)
            memory = torch.stack(cat_tensor)
            # Cross-Attention
            memory_k = memory + pos if self.pos_enc_at_cross_attn_keys else memory
            tgt = rearrange(tgt, 'b t p c -> b (t p) c')
            memory  = rearrange(memory, 'b t p c -> b (t p) c')
            memory_k  = rearrange(memory_k, 'b t p c -> b (t p) c')
            attnmask = torch.full((tgt.size(0), tgt.size(1), memory.size(1)), -float("Inf"), device=tgt.device, dtype=tgt.dtype)
            for i in range(bs): # b = 16
                for j in range(clip_T) : # t = 8 (cat mask token)
                    attnmask[i, j*197:j*197+197, :j*197+197] = 0            
            mask2 = torch.zeros(attnmask.size(1), attnmask.size(2)).view(-1).cuda()
            mask2[:int(len(mask2)*0.25)] = -float("Inf")
            mask2 = mask2[torch.randperm(len(mask2))]
            mask2 = mask2.view(attnmask.size(1), attnmask.size(2))
            mask2 = mask2.unsqueeze(0).repeat(attnmask.size(0),1,1)
            attnmask = attnmask+mask2
        else:
            memory_k =memory + pos if self.pos_enc_at_cross_attn_keys else memory
            tgt = rearrange(tgt, '(b t) p c -> b (t p) c', t=clip_T)
            memory  = rearrange(memory, '(b t) p c -> b (t p) c', t=clip_T)
            memory_k  = rearrange(memory_k, '(b t) p c -> b (t p) c', t=clip_T)
            attnmask = torch.full((tgt.size(1), memory.size(1)), float("0"), device=tgt.device, dtype=tgt.dtype)
        tgt2 = self.norm2(tgt)
        tgt2,attn = self.cross_attn_image(
            q=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
            k=memory_k,
            v=memory,
            attn_mask = attnmask,
            **kwds,
        )
        tgt = tgt + self.dropout2(tgt2)
        if mask is not None:
            tgt = rearrange(tgt, 'b (t p) c -> b t p c', t=clip_T)
            tgt = tgt[mask>-1]
        else:
            tgt = rearrange(tgt, 'b (t p) c -> (b t) p c', t=clip_T)

        return tgt,attn
    def forward(
        self,
        tgt,
        memory,
        mask = None, 
        mask_ID = None, 
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        num_k_exclude_rope: int = 0,
        clip_T = 4, 
        bs = 4
    ) -> torch.Tensor:
        tgt,attn = self._forward_sa(tgt, query_pos)
        tgt,attn = self._forward_ca(tgt, memory, query_pos, pos, mask, mask_ID, num_k_exclude_rope, clip_T=clip_T, bs=bs)
        # MLP
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt,attn
    
'''BaseLine only BEiT Model'''
class BEiT_model(nn.Module):
    def __init__(self, 
        num_class=2,
        **kwargs):
        super(BEiT_model,self).__init__()
        self.num_class = num_class
        feature_dim = 768
        self.backbone_model = modeling_finetune.beit_base_patch16_224(
            num_classes = 1000, drop_rate=0.0, drop_path_rate=0.1, attn_drop_rate=0.0,use_mean_pooling=False,
            init_scale = 0.001, init_values = 0.1, qkv_bias = True,
            use_abs_pos_emb=False, use_rel_pos_bias=True, use_shared_rel_pos_bias=False)
        # pretrain_path = './checkpoint-299.pth'
        # checkpoint = torch.load(pretrain_path, map_location='cpu')
        # checkpoint_model = checkpoint['model'] # module
        # msg = self.backbone_model.load_state_dict(checkpoint_model, strict=False)
        self.backbone_model.head = nn.Identity()
        self.backbone_model.norm = nn.Identity()
        self.normal = nn.LayerNorm(feature_dim*1)
        self.header = nn.Sequential(nn.Linear(feature_dim*1, num_class))
    def forward(self, x, landmark=None, eval=True):
        if eval == True:
            B, N, T, C, H, W = x.size()
            x = x.flatten(0,2)
            clstoken = self.backbone_model(x)
            output = self.normal(clstoken)
            output = self.header(output)
            output = output.view(B,N*T,-1)
            return output
        else:
            B, T, C, H, W = x.size()
            x = x.flatten(0,1)
            clstoken = self.backbone_model(x)
            output = self.normal(clstoken)
            output = self.header(output)
            output = output.view(B,T,-1)
            loss = torch.tensor(0).cuda()
            return output, loss


class BEiT(nn.Module):
    def __init__(self, 
        num_class=2,
        **kwargs):
        super(BEiT,self).__init__()
        self.num_class = num_class
        feature_dim = 768
        self.backbone_model = modeling_finetune.beit_base_patch16_224(
            num_classes = 1000, drop_rate=0.0, drop_path_rate=0.1, attn_drop_rate=0.0,use_mean_pooling=False,
            init_scale = 0.001, init_values = 0.1, qkv_bias = True,
            use_abs_pos_emb=False, use_rel_pos_bias=True, use_shared_rel_pos_bias=False)
        # pretrain_path = './checkpoint-299.pth'
        # checkpoint = torch.load(pretrain_path, map_location='cpu')
        # checkpoint_model = checkpoint['model'] # module
        # msg = self.backbone_model.load_state_dict(checkpoint_model, strict=False)
        self.backbone_model.head = nn.Identity()
        # self.backbone_model.norm = nn.Identity()
        self.norm_for_cls = nn.LayerNorm(feature_dim*1)
        self.header = nn.Sequential(nn.Linear(feature_dim*1, num_class))
    def forward(self, x,  videomask=None, maskID=None, landmark=None, eval=True):
        if eval:
            B, N, T, C, H, W = x.size()
            clstoken = self.backbone_model(x.flatten(0,2))
            output = self.norm_for_cls(clstoken)
            output = self.header(output)
            output = output.view(B,N*T,-1)
            return output
        else:
            if videomask is None:
                '''定长'''
                if len(x.size())==5:
                    B, T, C, H, W = x.size()
                    x = x.flatten(0,1)
                    N = 1
                else:
                    B, N, T, C, H, W = x.size()
                    x = x.flatten(0,2)                    
                clstoken = self.backbone_model(x)
                output = self.norm_for_cls(clstoken)
                output = self.header(output)
                output = output.view(B,N*T,-1)
                cosloss = torch.tensor(0).cuda()
            else:
                '''不定长'''
                B, T, C, H, W = x.size()
                x = x[videomask>-1]
                clstoken = self.backbone_model(x)
                output = self.norm_for_cls(clstoken)
                output = self.header(output)
                cosloss = torch.tensor(0).cuda()
            return output,cosloss

import models.lib.sam2 as sam2        
from models.lib.sam2.modeling.sam.prompt_encoder import PromptEncoder
from models.lib.sam2.modeling.sam.mask_decoder_custom import MaskDecoder_custom
from models.lib.sam2.modeling.sam.transformer import TwoWayTransformer
from models.lib.sam2.modeling.memory_attention_custom import MemoryAttentionLayer_clip, MemoryAttention_custom, MemoryAttention_onlySA,MemoryAttentionLayer_SA
from models.lib.sam2.modeling.sam.prompt_encoder import PromptEncoder
import models.util as model_util




class TFCU_Model_Stage1(nn.Module): 
    def __init__(self,  **kwargs):
        super(TFCU_Model_Stage1, self).__init__()
        '''Stage1'''
        self.Img_model = BEiT_model()
        self.Img_model.header = nn.Identity()
        self.mem_Attn = nn.ModuleList([MemoryAttentionLayer_custom(d_model=768, dim_feedforward=2048, dropout=0.1, pos_enc_at_attn=False, pos_enc_at_cross_attn_keys=True, pos_enc_at_cross_attn_queries=False) for i in range(4)] )
        self.pos_embed = nn.Parameter(torch.zeros(1, 197, 768))
        self.cosloss = torch.nn.CosineSimilarity(dim=-1)
        del self.Img_model.normal
        del self.Img_model.header
        self.mem_Encoder = nn.Identity()
        self.normal = nn.LayerNorm(768)
        self.header = nn.Sequential(nn.Linear(768, 2))
    def load_sd(self,path):
        checkpoint2 = torch.load(path, map_location='cpu')
        sd2 = checkpoint2['state_dict']
        prefix = 'module.'
        new_weights = {}
        for key, value in sd2.items():
            if key.startswith(prefix):
                new_key = key[len(prefix):]
                new_weights[new_key] = value
        msg = self.Img_model.load_state_dict(new_weights, strict=False)
        print(msg)
    def forward(self, video, videomask=None, maskID=None, eval=False, landmark=None):
        if eval:
            B, N, T, _, _, _ = video.size()
            forgery_feature, _, _ = self.Img_model.backbone_model.forward_features(video.flatten(0,2),return_all_tokens=True)
            mem_fea  = forgery_feature
            for i, blk in enumerate(self.mem_Attn):
                mem_fea, attn = blk(mem_fea, forgery_feature, pos=self.pos_embed,mask=videomask, mask_ID=maskID)
            pred_video = self.header(self.normal(mem_fea[:, 0]))
            pred_video = pred_video.view(B, N*T ,-1)
            return pred_video
        else:
            B, T, _, _, _ = video.size() # ([B, 8, 3, 224, 224])
            video_train = video[videomask>-1]    # ([B*8*0.5, 3, 224, 224])
            forgery_feature, _, _ = self.Img_model.backbone_model.forward_features(video_train,return_all_tokens=True)
            mem_fea  = forgery_feature
            for i, blk in enumerate(self.mem_Attn):
                mem_fea, attn = blk(mem_fea, forgery_feature, pos=self.pos_embed, clip_T = T, bs = B)
            pred_video = self.header(self.normal(mem_fea[:, 0]))
            return pred_video
        


class TFCU_Model(nn.Module): 
    def __init__(self,  **kwargs):
        super(TFCU_Model, self).__init__()
        '''Stage1'''
        self.Img_model = BEiT_model()
        self.Img_model.header = nn.Identity()
        self.mem_Attn = nn.ModuleList([MemoryAttentionLayer_custom(d_model=768, dim_feedforward=2048, dropout=0.1, pos_enc_at_attn=False, pos_enc_at_cross_attn_keys=True, pos_enc_at_cross_attn_queries=False) for i in range(4)] )
        self.pos_embed = nn.Parameter(torch.zeros(1, 197, 768))
        '''Stage2'''
        '''Memory_Attention'''
        self_attn = RoPEAttention(rope_theta=10000.0, feat_sizes=[32,32], num_heads=1,  downsample_rate=1, dropout=0.1, embedding_dim=768)
        cross_attn_image = RoPEAttention(rope_theta=10000.0, feat_sizes=[32,32], rope_k_repeat=True, num_heads=1, kv_in_dim=768, downsample_rate=1, dropout=0.1, embedding_dim=768)
        memory_layer = MemoryAttentionLayer_clip(self_attention=self_attn, cross_attention=cross_attn_image, activation="relu" ,d_model=768, dim_feedforward=2048, dropout=0.1, pos_enc_at_attn=False, pos_enc_at_cross_attn_keys=False, pos_enc_at_cross_attn_queries=False)
        self.memory_attention = MemoryAttention_custom(d_model=768, layer=memory_layer, num_layers=4, pos_enc_at_input=False, batch_first=False)
        '''Memory_Encoder'''
        self.position_encoding = sam2.modeling.position_encoding.PositionEmbeddingSine(num_pos_feats=768, normalize=True, scale=None, temperature=10000)
        self_attn = RoPEAttention(rope_theta=10000.0, feat_sizes=[32,32], num_heads=1,  downsample_rate=1, dropout=0.1, embedding_dim=768)
        cross_attn_image = RoPEAttention(rope_theta=10000.0, feat_sizes=[32,32], rope_k_repeat=True, num_heads=1, kv_in_dim=768, downsample_rate=1, dropout=0.1, embedding_dim=768)
        memory_layer = MemoryAttentionLayer_clip(self_attention=self_attn, cross_attention=cross_attn_image, activation="relu" ,d_model=768, dim_feedforward=2048, dropout=0.1, pos_enc_at_attn=False, pos_enc_at_cross_attn_keys=False, pos_enc_at_cross_attn_queries=False)
        self.memory_encoder = MemoryAttention_custom(d_model=768, layer=memory_layer, num_layers=4, pos_enc_at_input=False, batch_first=False)
        '''Init_State'''
        self.inference_state = self.init_state()
        for name, param in self.Img_model.named_parameters():
            param.requires_grad=False
        for name, param in self.mem_Attn.named_parameters():
            param.requires_grad=False
        self.cosloss = torch.nn.CosineSimilarity(dim=-1)
        self.normal2 = nn.LayerNorm(768*2)
        self.header2 = nn.Sequential(nn.Linear(768*2, 2))
        self.predicetor = MaskDecoder_custom(transformer=TwoWayTransformer(depth=2, embedding_dim=768, mlp_dim=2048, num_heads=8,))
        self.promptencoder = PromptEncoder(embed_dim=768, image_embedding_size=(14,14), input_image_size=(224,224), mask_in_chans=16)
        self.promptencoder.no_mask_embed.weight.requires_grad = False
        del self.promptencoder.mask_downscaling
        self.conv1 = nn.Conv2d(1536, 768, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.InstanceNorm2d(768)
        self.actv1 = nn.ReLU(inplace=True)
        self.Resblocks = model_util.ResBlocks(num_blocks=2, dim=768, norm='in', activation='relu', pad_type='zero')
    def init_state(self):
        inference_state = {}
        inference_state["encode_mem_feature"] = []
        inference_state["maskmem_pos_enc"] = []
        inference_state["forgery_feature"] = []
        inference_state["temporal_feature"] = []
        inference_state["memoryattn_feature"] = []
        inference_state["memoryattn_feature_longterm"] = []
        inference_state["prompt_embeding"] = []
        return inference_state
    def cycle(self, idx): # HRM
        windows_feature_t = self.inference_state["temporal_feature"][idx-1:idx] # n b t p c
        windows_feature_t = torch.stack(windows_feature_t) # 4 b t p c
        windows_feature_t = torch.mean(windows_feature_t, dim=0)# b t p c
        windows_feature_t = torch.mean(windows_feature_t, dim=1, keepdim=True)# b 1 p c
        for i in range(idx-1):
            w = 2/(idx-i)
            self.inference_state["temporal_feature"][i] = (1-w)*self.inference_state["temporal_feature"][i] + w*windows_feature_t
    def get_memory_feature(self): # MemoryBank
        encode_mem_feature = self.inference_state["encode_mem_feature"]
        prompt_mem_feature = self.inference_state["prompt_embeding"] # n B 5 2
        encode_mem_feature = rearrange(torch.stack(encode_mem_feature), 'n b p c -> b n p c')
        if encode_mem_feature.size(1)==1:
            copy_memory = encode_mem_feature
        else:
            copy_memory = encode_mem_feature.clone()
            copy_memory = torch.cat([copy_memory[:,:-1], encode_mem_feature[:,-1].unsqueeze(1)], dim=1)
        if copy_memory.size(1)>4:
            copy_memory = copy_memory[:,copy_memory.size(1)-4:]
            prompt_mem_feature = prompt_mem_feature[len(prompt_mem_feature)-4:]
        return copy_memory, prompt_mem_feature
    def Inference_single_frame(self, idx, fea, fea_ca, landmark): 
        ''' landmark: B N T 5 2 '''
        current_landmark = landmark[:, idx]
        current_vision_feats = fea[:,idx]
        current_fogery_feats = fea_ca[:,idx] # ''' landmark: B T 5 2 '''
        current_feature_tdn = current_vision_feats[:,current_vision_feats.size(1)//2]+torch.sum(current_vision_feats[:,1:]-current_vision_feats[:,:-1], dim=1) # b p c
        landmark_key = current_landmark[:,current_landmark.size(1)//2]  # ''' landmark: B 5 2 '''
        if idx == 0:
            feature_withmemory = current_vision_feats
        else:
            mask_memory, prompt_mem = self.get_memory_feature() # N B 5 2
            feature_sub = current_feature_tdn.unsqueeze(1)- mask_memory
            prompt_mem = torch.stack(prompt_mem)  # N B 5 2
            landmark_shift = prompt_mem - landmark_key.unsqueeze(0)  # N,B,5,2
            N,B,_,_ = landmark_shift.shape
            point_label = torch.ones_like(landmark_shift)[:,:,:,0]
            shift_point = (landmark_shift.flatten(0,1), point_label.flatten(0,1))
            landmark_shift_emd = self.promptencoder(points=shift_point,boxes=None, masks=None)[0]
            landmark_shift_emd = rearrange(landmark_shift_emd, '(b n) p c -> b n p c',n=N, b=B)
            landmark_shift_emd = torch.mean(landmark_shift_emd[:,:,1:], dim=2, keepdim=True)  # b n 1 c
            feature_sub = rearrange(feature_sub[:,:,1:], 'b n (h w) c -> b n h w c',h=14, w=14)
            landmark_shift_emd = landmark_shift_emd.unsqueeze(3).repeat(1,1,14,14,1)
            feature_sub_withlshiht = torch.cat([feature_sub, landmark_shift_emd], dim=-1)
            feature_sub_withlshiht = rearrange(feature_sub_withlshiht, 'b n h w c -> (b n) c h w')
            feature_sub_withlshiht = self.conv1(feature_sub_withlshiht)
            feature_sub_withlshiht = self.Resblocks(feature_sub_withlshiht)
            feature_sub_withlshiht = rearrange(feature_sub_withlshiht, '(b n) c h w -> b n (h w) c',b=B, n=N)
            feature_sub_withlshiht = torch.mean(feature_sub_withlshiht,dim=2, keepdim=True)
            mask_memory[:,:,1:] = mask_memory[:,:,1:] + feature_sub_withlshiht
            mask_memory =  rearrange(mask_memory, 'b n p c -> b (n p) c')
            feature_withmemory = self.memory_attention(current_fogery_feats, mask_memory)
        self.inference_state["memoryattn_feature"].append(feature_withmemory)
        ''' TDN '''
        forgery_fea, temporal_fea = self.predicetor(query = current_fogery_feats, key = feature_withmemory)
        self.inference_state["forgery_feature"].append(forgery_fea)
        self.inference_state["temporal_feature"].append(temporal_fea)
        current_feature_tdn = current_vision_feats[:,current_vision_feats.size(1)//2]+torch.sum(current_vision_feats[:,1:]-current_vision_feats[:,:-1], dim=1) # b p c
        feature_withmemory_tdn = feature_withmemory[:,feature_withmemory.size(1)//2]+torch.sum(feature_withmemory[:,1:]-feature_withmemory[:,:-1], dim=1) # b p c
        encode_mem_feature = self.memory_encoder(current_feature_tdn.unsqueeze(1), feature_withmemory_tdn)
        self.inference_state["encode_mem_feature"].append(encode_mem_feature.squeeze(1))        
        self.inference_state["prompt_embeding"].append(landmark_key)
        if len(self.inference_state["temporal_feature"])>2:
            self.cycle(idx)
    def forward(self, video, landmark=None, eval=False, frame_batch_size=None):
        B, N, T, _, _, _ = video.size()
        flat_video = video.flatten(0, 2)
        if frame_batch_size is None or frame_batch_size <= 0:
            forgery_feature, _, _ = self.Img_model.backbone_model.forward_features(
                flat_video.to(self.pos_embed.device), return_all_tokens=True)
        else:
            feature_chunks = []
            for frame_batch in flat_video.split(frame_batch_size, dim=0):
                features, _, _ = self.Img_model.backbone_model.forward_features(
                    frame_batch.to(self.pos_embed.device), return_all_tokens=True)
                feature_chunks.append(features)
            forgery_feature = torch.cat(feature_chunks, dim=0)
        mem_fea = forgery_feature
        for i, blk in enumerate(self.mem_Attn): # CCM
            mem_fea, attn = blk(mem_fea, forgery_feature, pos=self.pos_embed, clip_T = T, bs = B)
        forgery_feature_ca = mem_fea
        forgery_feature_ca = rearrange(forgery_feature_ca,   '(b n t) p c-> b n t p c', b=B, t=T, n=N)
        forgery_feature = rearrange(forgery_feature, '(b n t) p c-> b n t p c', b=B, t=T, n=N)
        '''Stage 2 '''
        B,N,T,_,_ = landmark.shape
        point_label = torch.ones_like(landmark)[:,:,:,:,0]
        points = (landmark.flatten(0,2), point_label.flatten(0,2))
        prompt_embding = self.promptencoder(points=points,boxes=None, masks=None)[0]
        prompt_embding = rearrange(prompt_embding, '(b n t) p c-> b n t p c', b= B, n=N, t=T)
        for frame_id in range(N): #LGM的主体
            self.Inference_single_frame(frame_id, forgery_feature, forgery_feature_ca, landmark)
        Final_feature1 = torch.stack(self.inference_state["forgery_feature"])
        Final_feature2 = torch.stack(self.inference_state["temporal_feature"])
        Final_feature1 = rearrange(Final_feature1, 'n b t p c-> b (n t) p c') 
        Final_feature2 = rearrange(Final_feature2, 'n b t p c-> b (n t) p c')
        
        Final_feature1 = Final_feature1[:,:,0]
        Final_feature2 = Final_feature2[:,:,0]
        Final_feature = torch.cat([Final_feature1, Final_feature2], dim=-1) # 特征融合部分

        predict = self.header2(self.normal2(Final_feature))
        self.inference_state = self.init_state()
        return predict

