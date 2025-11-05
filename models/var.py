import math
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

import dist
from models.basic_var import AdaLNBeforeHead, AdaLNSelfAttn
from models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from models.vqvae import VQVAE, VectorQuantizer2

from models.mamba.mixer_seq_simple import MambaLMHeadModel
from models.mamba.config_mamba import MambaConfig

class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)   # B16C


class VAR(nn.Module):
    def __init__(
        self, vae_local: VQVAE,
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True,
    ):
        super().__init__()
        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size
        self.depth = depth
        self.C = self.D = embed_dim
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.cond_drop_rate = cond_drop_rate

        # multi-scale meta
        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn*pn for pn in patch_nums)
        self.first_l = patch_nums[0]*patch_nums[0]
        self.begin_ends = []
        cur = 0
        for pn in patch_nums:
            self.begin_ends.append((cur, cur+pn*pn))
            cur += pn*pn
        self.num_stages_minus_1 = len(patch_nums)-1
        self.prog_si = -1
        self.rng = torch.Generator(device=dist.get_device())

        # quant & embeddings
        quant: VectorQuantizer2 = vae_local.quantize
        self.vae_proxy = (vae_local,)
        self.vae_quant_proxy = (quant,)
        self.word_embed = nn.Linear(self.Cvae, self.C)

        init_std = math.sqrt(1/self.C/3)
        self.class_emb = nn.Embedding(self.num_classes+1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight, std=init_std)
        self.uniform_prob = torch.full((1, num_classes), 1/num_classes, device=dist.get_device())

        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start, std=init_std)

        pos_1LC = []
        for pn in patch_nums:
            pe = torch.empty(1, pn*pn, self.C)
            nn.init.trunc_normal_(pe, std=init_std)
            pos_1LC.append(pe)
        self.pos_1LC = nn.Parameter(torch.cat(pos_1LC, dim=1))
        self.lvl_embed = nn.Embedding(len(patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight, std=init_std)

        self.register_buffer('lvl_1L',
            torch.cat([torch.full((pn*pn,), i) for i,pn in enumerate(patch_nums)]).view(1,self.L),
            persistent=False)

        # shared ada (可保留以兼容 cond 结构, 实际 MambaLMHeadModel 可在需要时使用)
        self.shared_ada_lin = nn.Sequential(nn.SiLU(), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()

        # head
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)

        # Mamba backbone 
        mcfg = MambaConfig(
            d_model=embed_dim,
            n_layer=depth,
            d_intermediate=int(embed_dim*mlp_ratio),
            vocab_size=self.V,
            num_classes=num_classes,
            num_tokens=self.L
        )
        self.mamba = MambaLMHeadModel(mcfg)

    def forward(self, label_B: torch.LongTensor, x_BLCv_wo_first_l: torch.Tensor):
        # 与原接口兼容：x_BLCv_wo_first_l 是 (B, L-first_l, Cvae) 的连续 teacher-forcing 向量
        B = x_BLCv_wo_first_l.shape[0]
        from torch import amp
        with amp.autocast('cuda', enabled=False):
            drop_mask = torch.rand(B, device=label_B.device) < self.cond_drop_rate
            label_eff = torch.where(drop_mask, self.num_classes, label_B)
            sos = self.class_emb(label_eff).unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start
            if self.prog_si == 0:
                seq = sos
            else:
                seq = torch.cat([sos, self.word_embed(x_BLCv_wo_first_l.float())], dim=1)  # B,L,C
            ed = self.begin_ends[self.prog_si][1] if self.prog_si >= 0 else self.L
            lvl = self.lvl_embed(self.lvl_1L[:, :ed]).expand(B, -1, -1)
            pos = self.pos_1LC[:, :ed]
            seq = seq[:, :ed] + lvl + pos  # B,ed,C

        # Mamba 前向：假设其 forward 支持 inputs_embeds (若当前实现只有 input_ids，需要在 MambaLMHeadModel 中加分支)
        outputs = self.mamba(inputs_embeds=seq, cond=label_eff)
        h = outputs.last_hidden_state  # B,ed,C
        logits = self.head(self.head_nm(h.float(), self.class_emb(label_eff)))
        return logits  # B,ed,V

    @torch.no_grad()
    def autoregressive_infer_cfg(self, B, label_B=None, g_seed=None, cfg=1.5, top_k=0, top_p=0.0, more_smooth=False):
        if g_seed is not None: self.rng.manual_seed(g_seed)
        rng = self.rng if g_seed is not None else None
        if label_B is None:
            label_B = torch.multinomial(self.uniform_prob, num_samples=B, generator=rng).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full((B,), label_B if label_B >= 0 else self.num_classes, device=self.pos_1LC.device)

        # CFG: 正样本 + 空类别
        label_2B = torch.cat([label_B, torch.full_like(label_B, self.num_classes)], dim=0)
        sos = self.class_emb(label_2B).unsqueeze(1).expand(2*B, self.first_l, -1) + self.pos_start
        lvl_all = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        seq = sos + lvl_all[:, :self.first_l]  # 2B, first_l, C

        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        cur_L = self.first_l

        for si, pn in enumerate(self.patch_nums):
            if si > 0:
                # 送入 Mamba 得到到目前为止所有位置的隐藏
                out = self.mamba(inputs_embeds=seq, cond=label_2B)
                h = out.last_hidden_state  # 2B,cur_L,C
                logits_stage = self.head(self.head_nm(h.float(), self.class_emb(label_2B)))[:, -pn*pn:, :]  # 2B, l_stage, V
                ratio = si / self.num_stages_minus_1
                t = cfg * ratio
                logits = (1+t)*logits_stage[:B] - t*logits_stage[B:]
                # 采样
                idx_Bl = sample_with_top_k_top_p_(logits, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]

                if not more_smooth:
                    h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)  # B,l,Cvae
                else:
                    gum_t = max(0.27*(1-ratio*0.95), 0.005)
                    h_BChw = gumbel_softmax_with_rng(logits.mul(1+ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
                h_BChw = h_BChw.transpose(1,2).reshape(B, self.Cvae, pn, pn)
                f_hat, next_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)

                if si != self.num_stages_minus_1:
                    # 准备下一阶段的输入 token embeddings
                    next_emb = self.word_embed(next_map.view(B, self.Cvae, -1).transpose(1,2))  # B,next_l,C
                    next_emb = next_emb + lvl_all[:, cur_L:cur_L + self.patch_nums[si+1]**2]
                    # CFG 扩展
                    next_emb = next_emb.repeat(2,1,1)
                    seq = torch.cat([seq, next_emb], dim=1)
                    cur_L += pn*pn

        return self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)

    def extra_repr(self):
        return 'mamba_backbone=1'

    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02, conv_std_or_gain=0.02):
        # 与 transformer 版保持风格一致，但去掉 self.blocks 相关
        if init_std < 0:
            init_std = (1 / self.C / 3) ** 0.5
        print(f'[init_weights] {type(self).__name__} with {init_std=:g}')
        for m in self.modules():
            with_w = hasattr(m, 'weight') and m.weight is not None
            with_b = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_((m.weight), std=init_std)
                if with_b: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=init_std)
                if getattr(m, 'padding_idx', None) is not None:
                    idx = m.padding_idx
                    if idx >= 0: m.weight.data[idx].zero_()
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm, nn.GroupNorm)):
                if with_w: m.weight.data.fill_(1.)
                if with_b: m.bias.data.zero_()
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d)):
                if conv_std_or_gain > 0:
                    nn.init.trunc_normal_(m.weight, std=conv_std_or_gain)
                else:
                    nn.init.xavier_normal_(m.weight, gain=-conv_std_or_gain)
                if with_b: m.bias.data.zero_()
        # 额外缩放 head
        if init_head >= 0 and isinstance(self.head, nn.Linear):
            self.head.weight.data.mul_(init_head)
            if self.head.bias is not None:
                self.head.bias.data.zero_()
        # AdaLNBeforeHead 的自适应线性
        if isinstance(self.head_nm, AdaLNBeforeHead) and hasattr(self.head_nm, 'ada_lin'):
            al = self.head_nm.ada_lin[-1]
            if hasattr(al, 'weight'):
                al.weight.data.mul_(init_adaln)
                # 模仿 transformer：前 2*self.C 通常是 shift/scale 的微小初始化
                if al.weight.shape[0] >= 2*self.C:
                    al.weight.data[:2*self.C].mul_(init_adaln_gamma)
            if hasattr(al, 'bias') and al.bias is not None:
                al.bias.data.zero_()

class VARHF(VAR, PyTorchModelHubMixin):
            # repo_url="https://github.com/FoundationVision/VAR",
            # tags=["image-generation"]):
    def __init__(
        self,
        vae_kwargs,
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True,
    ):
        vae_local = VQVAE(**vae_kwargs)
        super().__init__(
            vae_local=vae_local,
            num_classes=num_classes, depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            norm_eps=norm_eps, shared_aln=shared_aln, cond_drop_rate=cond_drop_rate,
            attn_l2_norm=attn_l2_norm,
            patch_nums=patch_nums,
            flash_if_available=flash_if_available, fused_if_available=fused_if_available,
        )
