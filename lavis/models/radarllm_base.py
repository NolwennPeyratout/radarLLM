"""
Radar-first shared components for RadarLLM models.
"""

import contextlib
import logging
import os
from typing import Optional

import torch
import torch.nn as nn
from transformers import BertTokenizer

from lavis.common.dist_utils import download_cached_file
from lavis.common.utils import is_url
from lavis.models.base_model import BaseModel
from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel
from lavis.models.voxelnet import VoxelNet


class RadarLLMBase(BaseModel):
    """Shared radar/QFormer utilities used by pretrain and downstream models."""

    def __init__(
        self,
        Z_rad: int,
        Y_rad: int,
        X_rad: int,
        latent_dim: int = 128,
        use_radar_occupancy_map: bool = False,
        use_rpn_radar: bool = False,
        num_query_token: int = 32,
        max_txt_len: int = 128,
        qformer_text_input: bool = True,
    ):
        super().__init__()

        self.Z_rad, self.Y_rad, self.X_rad = Z_rad, Y_rad, X_rad
        self.latent_dim = latent_dim
        self.max_txt_len = max_txt_len
        self.qformer_text_input = qformer_text_input

        self.tokenizer = self.init_tokenizer(truncation_side="left")

        self.radar_encoder = VoxelNet(
            use_col=use_rpn_radar,
            reduced_zx=False,
            output_dim=latent_dim,
            use_radar_occupancy_map=use_radar_occupancy_map,
        )

        self.Qformer, self.query_tokens = self.init_Qformer(num_query_token, latent_dim)

        if not qformer_text_input:
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            self.Qformer.resize_token_embeddings(len(self.tokenizer))

    @classmethod
    def init_Qformer(cls, num_query_token: int, vision_width: int, cross_attention_freq: int = 2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        qformer = BertLMHeadModel.from_pretrained("bert-base-uncased", config=encoder_config)
        query_tokens = nn.Parameter(torch.zeros(1, num_query_token, encoder_config.hidden_size))
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return qformer, query_tokens

    @classmethod
    def init_tokenizer(cls, truncation_side: str = "right"):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side=truncation_side)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer

    def maybe_autocast(self, dtype=torch.float16):
        enable_autocast = self.device != torch.device("cpu")
        if enable_autocast:
            return torch.amp.autocast("cuda", dtype=dtype)
        return contextlib.nullcontext()

    def _encode_radar(self, rad_occ_mem0, device: torch.device) -> torch.Tensor:
        radar_device = next(self.radar_encoder.parameters()).device

        assert isinstance(rad_occ_mem0, (tuple, list)) and len(rad_occ_mem0) == 3, (
            f"Expected rad_occ_mem0 to be a tuple/list of (voxel_features, voxel_coords, num_occupied), "
            f"got {type(rad_occ_mem0)}"
        )

        voxel_features = rad_occ_mem0[0].to(radar_device, non_blocking=True)
        voxel_coords = rad_occ_mem0[1].to(radar_device, non_blocking=True)
        number_of_occupied_voxels = rad_occ_mem0[2].to(radar_device, non_blocking=True)

        rad_bev = self.radar_encoder(
            voxel_features=voxel_features,
            voxel_coords=voxel_coords,
            number_of_occupied_voxels=number_of_occupied_voxels,
        )

        if rad_bev.dim() != 4:
            raise ValueError(f"Expected radar BEV tensor with 4 dims (B, C, Z, X), got {tuple(rad_bev.shape)}")

        bsz, channels, z_dim, x_dim = rad_bev.shape
        rad_tokens = rad_bev.permute(0, 2, 3, 1).reshape(bsz, z_dim * x_dim, channels)
        return rad_tokens.to(device).contiguous()

    def load_from_pretrained(self, url_or_filename: str, weights_only: bool = True):
        if is_url(url_or_filename):
            cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
            checkpoint = torch.load(cached_file, map_location="cpu", weights_only=weights_only)
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu", weights_only=weights_only)
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        if "model_state_dict" in checkpoint:
            checkpoint["model"] = checkpoint.pop("model_state_dict")

        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        msg = self.load_state_dict(state_dict, strict=False)

        logging.info("load checkpoint from %s", url_or_filename)
        return msg

    def _apply_voxelnet_freeze_policy(self, should_freeze: bool):
        for _, param in self.radar_encoder.named_parameters():
            param.requires_grad = not should_freeze

    def load_checkpoint_from_config(self, cfg, **kwargs):
        load_finetuned = cfg.get("load_finetuned", True)
        if load_finetuned:
            finetune_path = cfg.get("finetuned", None)
            assert finetune_path is not None, "Found load_finetuned is True, but finetune_path is None."
            self.load_checkpoint(url_or_filename=finetune_path)
            return

        load_qformer_pretrained = cfg.get("load_qformer_pretrained", False)
        if load_qformer_pretrained:
            qformer_pretrain_path = cfg.get("qformer_pretrained", None)
            assert qformer_pretrain_path is not None, (
                "Found load_qformer_pretrained is True, but qformer_pretrain_path is None."
            )
            self.load_from_pretrained(url_or_filename=qformer_pretrain_path)

        load_voxelnet_pretrained = cfg.get("load_voxelnet_pretrained", False)
        if load_voxelnet_pretrained:
            voxelnet_pretrain_path = cfg.get("voxelnet_pretrained", None)
            assert voxelnet_pretrain_path is not None, (
                "Found load_voxelnet_pretrained is True, but voxelnet_pretrain_path is None."
            )
            self.load_from_pretrained(url_or_filename=voxelnet_pretrain_path, weights_only=False)

        load_pretrained = cfg.get("load_pretrained", True)
        if load_pretrained:
            pretrain_path = cfg.get("pretrained", None)
            assert pretrain_path is not None, "Found load_pretrained is True, but pretrain_path is None."
            self.load_from_pretrained(url_or_filename=pretrain_path, **kwargs)
