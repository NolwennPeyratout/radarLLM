"""

Models for RadarLLM, including the main model and its components.
radar encoder : voxel net from BEVCar
text encoder : from Qwen/Qwen2.5-3B Instruct
QFormer from BLIP2
LLM : Qwen/Qwen2.5-3B - Instruct
RadarLLM with VoxelNet + QFormer + Qwen2.5 CausalLM.

Sample contract mirrors blip2_t5_instruct:
- samples["text_input"]: list[str]
- samples["text_output"]: list[str] (training)
- samples["prompt"]: optional list[str] (generation)
- samples["rad_occ_mem0"]: Tensor (B, 16, Z, Y, X) ( C= 16 )
"""
import os
import contextlib
import logging
from datetime import datetime
from typing import List, Optional, Tuple, Dict
from peft import LoraConfig, get_peft_model

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BertTokenizer

from lavis.common.registry import registry
from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel
from lavis.models.blip2_models.blip2 import Blip2Base
from lavis.models.base_model import BaseModel
from lavis.models.voxelnet import VoxelNet
from lavis.common.dist_utils import download_cached_file
from lavis.common.utils import is_url
from lavis.common.annotator.uniformer.mmcv.utils.logging import get_logger, logger_initialized, print_log


@registry.register_model("radar_llm_qwen")
class RadarLLM(BaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "qwen2.5-3B-Instruct": "configs/models/radar_llm_qwen2_5_3b.yaml",
    }

    def __init__(
        self,
        Z_rad: int,
        Y_rad: int,
        X_rad: int,
        latent_dim: int = 128,
        use_radar_occupancy_map: bool = False,
        use_rpn_radar: bool = False,
        num_query_token: int = 32,
        llm_model: str = "/home/renault/repo/models/Qwen2.5-3B-Instruct",
        prompt: str = "",
        max_txt_len: int = 128,
        max_output_txt_len: int = 256,
        apply_lemmatizer: bool = False,
        qformer_text_input: bool = True,
        has_lora: bool = False,
        save_radar_bev_debug: bool = False,
        radar_bev_debug_dir: str = "outputs/radar_bev_debug",
        radar_bev_debug_max_items: int = 1,
        use_chat_template: bool = True,
        include_system_prompt: bool = True,
        system_prompt: str = "You are a helpful assistant.",
        use_radar_token: bool = True,
        radar_token: str = "<radar>",
    ):
        super().__init__()

        self.Z_rad, self.Y_rad, self.X_rad = Z_rad, Y_rad, X_rad
        self.latent_dim = latent_dim
        self.max_txt_len = max_txt_len
        self.max_output_txt_len = max_output_txt_len
        self.qformer_text_input = qformer_text_input
        self.prompt = prompt
        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None
        self.save_radar_bev_debug = save_radar_bev_debug
        self.radar_bev_debug_dir = radar_bev_debug_dir
        self.radar_bev_debug_max_items = max(1, int(radar_bev_debug_max_items))
        self._radar_bev_debug_step = 0
        self.use_chat_template = use_chat_template
        self.include_system_prompt = include_system_prompt
        self.system_prompt = system_prompt
        self.use_radar_token = use_radar_token
        self.radar_token = radar_token
        self.radar_token_id: Optional[int] = None
        self._chat_train_debug_logged = True #Set false to enable debug logs for chat template training tokenization and labeling
        self._chat_infer_debug_logged = True #Set false to enable debug logs for inference as well

        # Get the initialized logger, if not exist,
        # create a logger named `mmcv`
        logger_names = list(logger_initialized.keys())
        self.logger_name = logger_names[0] if logger_names else 'mmcv'

        #Initialization tokenizer for text for the QFormer
        self.tokenizer = self.init_tokenizer(truncation_side="left")

        #Radar encoder:
        #Need to check the truthness of the latent dim because its qformer after.
        # if reduced_zx==True -> 100x100 instead of 200x200
        # if use_col=True: added RPN after CML 
        self.radar_encoder = VoxelNet(
            use_col=use_rpn_radar,
            reduced_zx=False,
            output_dim=latent_dim,
            use_radar_occupancy_map=use_radar_occupancy_map,
        )

        print_log("#############    VOXELNET RADAR ENCODING INITIALIZED   ##############", logger=self.logger_name)

        self.Qformer, self.query_tokens = self.init_Qformer(num_query_token, latent_dim)

        #If the text is not in the input of the QFormer
        #Not used I think for us now.
        #Could be used to check the difference.
        if not qformer_text_input:
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.Qformer.cls = None

        print_log("#############    QFORMER INITIALIZED   ##############", logger=self.logger_name)

        self.qwen_tokenizer = AutoTokenizer.from_pretrained(
            llm_model,
            truncation_side="left",
            use_fast=True,
        )
        self.qwen_output_tokenizer = AutoTokenizer.from_pretrained(
            llm_model,
            truncation_side="right",
            use_fast=True,
        )
        if self.qwen_tokenizer.pad_token is None:
            self.qwen_tokenizer.pad_token = self.qwen_tokenizer.eos_token
        if self.qwen_output_tokenizer.pad_token is None:
            self.qwen_output_tokenizer.pad_token = self.qwen_output_tokenizer.eos_token

        print_log("#############    QWEN TOKENIZERS INITIALIZED    ##############", logger=self.logger_name)

        self.qwen_model = AutoModelForCausalLM.from_pretrained(
            llm_model,
            torch_dtype=torch.float32,
        )

        self._initialize_radar_token()

        self.qwen_model.config.pad_token_id = self.qwen_tokenizer.pad_token_id
        if getattr(self.qwen_model, "generation_config", None) is not None:
            self.qwen_model.generation_config.pad_token_id = self.qwen_tokenizer.pad_token_id

        if has_lora:
            loraconfig = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj","v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.qwen_model = get_peft_model(self.qwen_model, loraconfig)
            self.qwen_model.print_trainable_parameters()

            # GradScaler expects trainable gradients to be fp32 when using AMP.
            # LoRA adapters can inherit fp16 dtype from the base model, so cast
            # only trainable parameters to float32 to avoid unscale errors.
            #for _, param in self.qwen_model.named_parameters():
            #    if param.requires_grad and param.dtype != torch.float32:
            #        param.data = param.data.float()
        else:
            for name, param in self.qwen_model.named_parameters():
                param.requires_grad = False

        print_log("#############    QWEN MODEL INITIALIZED    ##############", logger=self.logger_name)

        #The linear projection layer to align the Qformer output with the Qwen input

        self.qwen_proj = nn.Linear(
            self.Qformer.config.hidden_size,
            self.qwen_model.config.hidden_size,
        )

        print_log("Initialized RadarLLM with %s", llm_model, logger=self.logger_name)

    def _initialize_radar_token(self) -> None:
        if not self.use_radar_token:
            return

        # Keep tokenizer/model vocab in sync for radar placeholder replacement.
        num_new_tokens = self.qwen_tokenizer.add_tokens([self.radar_token], special_tokens=True)
        self.qwen_output_tokenizer.add_tokens([self.radar_token], special_tokens=True)
        self.qwen_model.resize_token_embeddings(len(self.qwen_tokenizer))

        self.radar_token_id = self.qwen_tokenizer.convert_tokens_to_ids(self.radar_token)
        output_token_id = self.qwen_output_tokenizer.convert_tokens_to_ids(self.radar_token)
        if output_token_id != self.radar_token_id:
            print_log(
                "Radar token id mismatch between tokenizers (train=%d, output=%d).",
                self.radar_token_id,
                output_token_id,
                logger=self.logger_name,
                level=logging.WARNING,
            )

        if num_new_tokens > 0:
            input_embeddings = self.qwen_model.get_input_embeddings().weight.data
            output_embeddings = self.qwen_model.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

    def _prepend_radar_token_if_missing(self, text: str) -> str:
        if not self.use_radar_token:
            return text
        if self.radar_token in text:
            return text
        stripped = text.strip()
        if stripped:
            return f"{self.radar_token} {stripped}"
        return self.radar_token

    def _prepend_radar_token_batch_if_missing(self, text_batch: List[str]) -> List[str]:
        return [self._prepend_radar_token_if_missing(text) for text in text_batch]

    def _replace_radar_token_with_prefix(
        self,
        token_ids: torch.Tensor,
        token_embeds: torch.Tensor,
        token_attention_mask: torch.Tensor,
        prefix_embeds: torch.Tensor,
        token_labels: Optional[torch.Tensor] = None,
        prepend_prefix_if_missing: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if self.radar_token_id is None:
            raise ValueError("Radar token id is not initialized.")

        batch_size = token_ids.size(0)
        embed_dim = token_embeds.size(-1)
        prefix_len = prefix_embeds.size(1)

        updated_embeds: List[torch.Tensor] = []
        updated_atts: List[torch.Tensor] = []
        updated_labels: List[torch.Tensor] = []

        for batch_idx in range(batch_size):
            valid_positions = torch.where(token_attention_mask[batch_idx] > 0)[0]
            seq_embeds = token_embeds[batch_idx, valid_positions]
            seq_ids = token_ids[batch_idx, valid_positions]
            seq_labels = token_labels[batch_idx, valid_positions] if token_labels is not None else None

            radar_positions = torch.where(seq_ids == self.radar_token_id)[0]
            if radar_positions.numel() > 0:
                radar_pos = int(radar_positions[0].item())
                new_embeds = torch.cat(
                    [
                        seq_embeds[:radar_pos],
                        prefix_embeds[batch_idx],
                        seq_embeds[radar_pos + 1 :],
                    ],
                    dim=0,
                )
                if seq_labels is not None:
                    new_labels = torch.cat(
                        [
                            seq_labels[:radar_pos],
                            torch.full(
                                (prefix_len,),
                                -100,
                                dtype=seq_labels.dtype,
                                device=seq_labels.device,
                            ),
                            seq_labels[radar_pos + 1 :],
                        ],
                        dim=0,
                    )
            elif prepend_prefix_if_missing:
                print_log("[RADAR_TOKEN] Radar token not found in sequence, prepending prefix." \
                " Batch idx: %d", batch_idx, logger=self.logger_name)
                new_embeds = torch.cat([prefix_embeds[batch_idx], seq_embeds], dim=0)
                if seq_labels is not None:
                    new_labels = torch.cat(
                        [
                            torch.full(
                                (prefix_len,),
                                -100,
                                dtype=seq_labels.dtype,
                                device=seq_labels.device,
                            ),
                            seq_labels,
                        ],
                        dim=0,
                    )
            else:
                new_embeds = seq_embeds
                if seq_labels is not None:
                    new_labels = seq_labels

            new_atts = torch.ones(
                (new_embeds.size(0),),
                dtype=token_attention_mask.dtype,
                device=token_attention_mask.device,
            )
            updated_embeds.append(new_embeds)
            updated_atts.append(new_atts)
            if token_labels is not None:
                updated_labels.append(new_labels)

        max_len = max(t.size(0) for t in updated_embeds)
        padded_embeds = []
        padded_atts = torch.zeros(
            (batch_size, max_len),
            dtype=token_attention_mask.dtype,
            device=token_attention_mask.device,
        )

        if token_labels is not None:
            padded_labels = torch.full(
                (batch_size, max_len),
                -100,
                dtype=token_labels.dtype,
                device=token_labels.device,
            )
        else:
            padded_labels = None

        for batch_idx, cur_embeds in enumerate(updated_embeds):
            cur_len = cur_embeds.size(0)
            if cur_len < max_len:
                cur_embeds = torch.cat(
                    [
                        cur_embeds,
                        torch.zeros(
                            (max_len - cur_len, embed_dim),
                            dtype=cur_embeds.dtype,
                            device=cur_embeds.device,
                        ),
                    ],
                    dim=0,
                )
            padded_embeds.append(cur_embeds)
            padded_atts[batch_idx, :cur_len] = 1
            if padded_labels is not None:
                padded_labels[batch_idx, :cur_len] = updated_labels[batch_idx]

        return torch.stack(padded_embeds, dim=0), padded_atts, padded_labels

    def _build_chat_messages(
        self,
        user_text: str,
        assistant_text: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if self.include_system_prompt and self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_text})
        if assistant_text is not None:
            messages.append({"role": "assistant", "content": assistant_text})
        return messages

    def _tokenize_chat_train(
        self,
        text_input: List[str],
        text_output: List[str],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not hasattr(self.qwen_tokenizer, "apply_chat_template"):
            raise ValueError("Tokenizer does not support apply_chat_template.")

        self.qwen_tokenizer.padding_side = "right"

        chat_texts: List[str] = []
        assistant_starts: List[int] = []
        debug_full_text = None
        debug_prompt_text = None
        for user_text, assistant_text in zip(text_input, text_output):
            user_text = self._prepend_radar_token_if_missing(user_text)
            full_messages = self._build_chat_messages(user_text, assistant_text)
            prompt_messages = self._build_chat_messages(user_text, assistant_text=None)

            full_text = self.qwen_tokenizer.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            prompt_text = self.qwen_tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            full_ids = self.qwen_tokenizer(
                full_text,
                add_special_tokens=False,
                return_tensors=None,
            )["input_ids"]
            prompt_ids = self.qwen_tokenizer(
                prompt_text,
                add_special_tokens=False,
                return_tensors=None,
            )["input_ids"]

            chat_texts.append(full_text)
            assistant_starts.append(min(len(prompt_ids), len(full_ids)))
            if debug_full_text is None:
                debug_full_text = full_text
                debug_prompt_text = prompt_text

        tokenized = self.qwen_tokenizer(
            chat_texts,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len + self.max_output_txt_len,
            return_tensors="pt",
        ).to(device)

        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask
        labels = input_ids.clone()
        labels = labels.masked_fill(attention_mask == 0, -100)

        for i, start in enumerate(assistant_starts):
            start = min(start, labels.shape[1])
            labels[i, :start] = -100

        # One-shot debug logs to verify chat template shape and assistant-only supervision.
        if not self._chat_train_debug_logged and labels.shape[0] > 0:
            self._chat_train_debug_logged = True
            sample_idx = 0
            sample_len = int(attention_mask[sample_idx].sum().item())
            assistant_start = min(assistant_starts[sample_idx], labels.shape[1])
            supervised_mask = labels[sample_idx] != -100
            supervised_total = int(supervised_mask[:sample_len].sum().item())
            supervised_before_assistant = int(supervised_mask[:assistant_start].sum().item())
            supervised_after_assistant = int(supervised_mask[assistant_start:sample_len].sum().item())

            print_log("[CHAT_DEBUG][TRAIN] Raw chat full (before tokenization): %s", debug_full_text, logger=self.logger_name)
            print_log("[CHAT_DEBUG][TRAIN] Raw chat prompt-only (before tokenization): %s", debug_prompt_text, logger=self.logger_name)
            print_log(
                "[CHAT_DEBUG][TRAIN] label stats sample0 -> seq_len=%d, assistant_start=%d, supervised_total=%d, supervised_before_assistant=%d, supervised_after_assistant=%d",
                sample_len,
                assistant_start,
                supervised_total,
                supervised_before_assistant,
                supervised_after_assistant,
                logger=self.logger_name
            )
            if supervised_before_assistant != 0:
                print_log(
                    "[CHAT_DEBUG][TRAIN] Expected zero supervised tokens before assistant span, got %d",
                    supervised_before_assistant,
                    logger=self.logger_name, 
                    level=logging.WARNING,
                )

        return input_ids, attention_mask, labels

    def _tokenize_chat_infer(
        self,
        text_input: List[str],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not hasattr(self.qwen_tokenizer, "apply_chat_template"):
            raise ValueError("Tokenizer does not support apply_chat_template.")

        self.qwen_tokenizer.padding_side = "left"

        prompt_texts: List[str] = []
        for user_text in text_input:
            user_text = self._prepend_radar_token_if_missing(user_text)
            prompt_messages = self._build_chat_messages(user_text, assistant_text=None)
            prompt_text = self.qwen_tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_texts.append(prompt_text)

        if not self._chat_infer_debug_logged and len(prompt_texts) > 0:
            self._chat_infer_debug_logged = True
            logging.info("[CHAT_DEBUG][INFER] Raw chat prompt (before tokenization): %s", prompt_texts[0])

        tokenized = self.qwen_tokenizer(
            prompt_texts,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(device)

        return tokenized.input_ids, tokenized.attention_mask

    @staticmethod
    def _apply_bev_colormap(norm_counts: torch.Tensor) -> torch.Tensor:
        """
        Convert normalized BEV counts in [0, 1] to RGB heatmap values in [0, 255].
        Input shape: (Z, X).
        Output shape: (Z, X, 3).
        """
        r = torch.clamp(2.0 * norm_counts - 0.5, min=0.0, max=1.0)
        g = torch.clamp(1.0 - torch.abs(2.0 * norm_counts - 1.0), min=0.0, max=1.0)
        b = torch.clamp(1.5 - 2.0 * norm_counts, min=0.0, max=1.0)
        rgb = torch.stack([r, g, b], dim=-1)
        return (rgb * 255.0).to(torch.uint8)

    def _save_radar_bev_debug(self, rad_bev: torch.Tensor, tag: str = "radar") -> None:
        """
        Save BEV debug images where color encodes how many channels are active per voxel.
        rad_bev shape: (B, C, Z, X).
        """
        if not self.save_radar_bev_debug:
            return

        try:
            from PIL import Image
        except ImportError:
            logging.warning("PIL is not available. Install pillow to enable radar BEV debug visualization.")
            return

        os.makedirs(self.radar_bev_debug_dir, exist_ok=True)

        with torch.no_grad():
            bev = rad_bev.detach().to("cpu")
            # Count non-zero feature channels for each BEV cell.
            counts = (bev.abs() > 1e-8).sum(dim=1).float()  # (B, Z, X)
            batch_items = min(counts.shape[0], self.radar_bev_debug_max_items)

            for b_idx in range(batch_items):
                count_map = counts[b_idx]
                max_val = torch.clamp(count_map.max(), min=1.0)
                norm_map = count_map / max_val
                rgb = self._apply_bev_colormap(norm_map)
                image = Image.fromarray(rgb.numpy(), mode="RGB")

                filename = (
                    f"{tag}_step{self._radar_bev_debug_step:06d}_b{b_idx}_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )
                path = os.path.join(self.radar_bev_debug_dir, filename)
                image.save(path)
                logging.info("Saved radar BEV debug image: %s", path)

            self._radar_bev_debug_step += 1

    def _encode_radar(self, rad_occ_mem0, device: torch.device, print_voxel=False) -> torch.Tensor:
        radar_device = next(self.radar_encoder.parameters()).device

        assert isinstance(rad_occ_mem0, (tuple, list)) and len(rad_occ_mem0) == 3 ,f"Expected rad_occ_mem0 to be a tuple/list of (voxel_features, voxel_coords, num_occupied), got {type(rad_occ_mem0)} with length {len(rad_occ_mem0) if isinstance(rad_occ_mem0, (tuple, list)) else 'N/A'}"
        
        # Keep sparse VoxelNet inputs on the same device as radar_encoder.
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

        #self._save_radar_bev_debug(rad_bev, tag="post_voxelnet")

        # QFormer cross-attention expects (B, N, C) tokens.
        b, cy, z, x = rad_bev.shape
        rad_tokens = rad_bev.permute(0, 2, 3, 1).reshape(b, z * x, cy)

        if print_voxel:
            if torch.is_tensor(rad_occ_mem0):
                input_shape = tuple(rad_occ_mem0.shape)
            elif isinstance(rad_occ_mem0, (tuple, list)) and len(rad_occ_mem0) == 3:
                input_shape = (
                    tuple(rad_occ_mem0[0].shape),
                    tuple(rad_occ_mem0[1].shape),
                    tuple(rad_occ_mem0[2].shape),
                )
            else:
                input_shape = "unknown"

            logging.info(
                "Radar input shape: %s, encoded BEV shape: %s, tokens shape: %s",
                input_shape,
                tuple(rad_bev.shape),
                tuple(rad_tokens.shape),
            )

            


        return rad_tokens.to(device)

    def _build_prefix(self, rad_occ_mem0, text_input: List[str], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        rad_tokens = self._encode_radar(rad_occ_mem0, device)
        radar_atts = torch.ones(rad_tokens.size()[:-1], dtype=torch.long, device=device)

        # Ensure query_tokens is on the same device as input
        #Expand query tokens to batch size
        query_tokens = self.query_tokens.to(device).expand(rad_tokens.size(0), -1, -1)
        if self.qformer_text_input:
            text_qformer = self.tokenizer(
                text_input,
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=device)
            qformer_atts = torch.cat([query_atts, text_qformer.attention_mask], dim=1)
            query_output = self.Qformer.bert(
                text_qformer.input_ids,
                attention_mask=qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=rad_tokens,
                encoder_attention_mask=radar_atts,
                return_dict=True,
            )
        else:
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=rad_tokens,
                encoder_attention_mask=radar_atts,
                return_dict=True,
            )

        prefix_embeds = self.qwen_proj(
            query_output.last_hidden_state[:, : query_tokens.size(1), :]
        )
        prefix_atts = torch.ones(prefix_embeds.size()[:-1], dtype=torch.long, device=device)
        return prefix_embeds, prefix_atts

    def forward(self, samples):
        rad_occ_mem0 = samples["rad_occ_mem0"]
        text_input = samples["text_input"]
        text_output = samples["text_output"]

        if isinstance(text_input, str):
            text_input = [text_input]
        if isinstance(text_output, str):
            text_output = [text_output]

        # Use model device as source of truth to avoid stale CPU inputs from dataloader side.
        device = next(self.qwen_model.parameters()).device

        with self.maybe_autocast(dtype=torch.float32):
            prefix_embeds, prefix_atts = self._build_prefix(rad_occ_mem0, text_input, device)

            if self.use_chat_template and hasattr(self.qwen_tokenizer, "apply_chat_template"):
                chat_input_ids, chat_attention_mask, chat_labels = self._tokenize_chat_train(
                    text_input=text_input,
                    text_output=text_output,
                    device=device,
                )

                chat_embeds = self.qwen_model.get_input_embeddings()(chat_input_ids)
                if self.use_radar_token and self.radar_token_id is not None:
                    full_embeds, full_atts, full_labels = self._replace_radar_token_with_prefix(
                        token_ids=chat_input_ids,
                        token_embeds=chat_embeds,
                        token_attention_mask=chat_attention_mask,
                        prefix_embeds=prefix_embeds,
                        token_labels=chat_labels,
                        prepend_prefix_if_missing=True,
                    )
                else:
                    full_embeds = torch.cat([prefix_embeds, chat_embeds], dim=1)
                    full_atts = torch.cat([prefix_atts, chat_attention_mask], dim=1)

                    prefix_ignore = torch.full(
                        (chat_labels.size(0), prefix_embeds.size(1)),
                        -100,
                        dtype=chat_labels.dtype,
                        device=device,
                    )
                    full_labels = torch.cat([prefix_ignore, chat_labels], dim=1)
            else:
                # Legacy path kept for backward compatibility.
                text_input = self._prepend_radar_token_batch_if_missing(text_input)
                input_tokens = self.qwen_tokenizer(
                    text_input,
                    padding="longest",
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(device)
                output_tokens = self.qwen_output_tokenizer(
                    text_output,
                    padding="longest",
                    truncation=True,
                    max_length=self.max_output_txt_len,
                    return_tensors="pt",
                ).to(device)

                # Embed prompt tokens and output tokens.
                input_embeds = self.qwen_model.get_input_embeddings()(input_tokens.input_ids)
                output_embeds = self.qwen_model.get_input_embeddings()(output_tokens.input_ids)

                # Start from token ids [prompt_tokens] + [output_tokens].
                # We use them as labels for teacher forcing.
                labels = torch.cat([input_tokens.input_ids, output_tokens.input_ids], dim=1)
                # We do NOT train on prompt tokens, so we set them to -100 (ignore index).
                labels[:, : input_tokens.input_ids.size(1)] = -100
                # masked_fill replaces values where mask == True.
                # Here we ignore padded positions in both prompt and output.
                llm_mask = torch.cat([input_tokens.attention_mask, output_tokens.attention_mask], dim=1)
                labels = labels.masked_fill(llm_mask == 0, -100)

                llm_embeds = torch.cat([input_embeds, output_embeds], dim=1)
                if self.use_radar_token and self.radar_token_id is not None:
                    llm_ids = torch.cat([input_tokens.input_ids, output_tokens.input_ids], dim=1)
                    full_embeds, full_atts, full_labels = self._replace_radar_token_with_prefix(
                        token_ids=llm_ids,
                        token_embeds=llm_embeds,
                        token_attention_mask=llm_mask,
                        prefix_embeds=prefix_embeds,
                        token_labels=labels,
                        prepend_prefix_if_missing=True,
                    )
                else:
                    # Build one causal sequence: [radar_prefix] + [prompt_tokens] + [output_tokens].
                    full_embeds = torch.cat([prefix_embeds, llm_embeds], dim=1)
                    full_atts = torch.cat([prefix_atts, llm_mask], dim=1)

                    # torch.full creates a tensor filled with one value.
                    # We create ignore labels for radar prefix tokens because they are not text targets.
                    prefix_ignore = torch.full(
                        (labels.size(0), prefix_embeds.size(1)),
                        -100,
                        dtype=labels.dtype,
                        device=device,
                    )
                    # Final labels match full_embeds length: [prefix_ignore] + [prompt/output labels].
                    full_labels = torch.cat([prefix_ignore, labels], dim=1)

            outputs = self.qwen_model(
                inputs_embeds=full_embeds,
                attention_mask=full_atts,
                labels=full_labels,
                return_dict=True,
            )

        return {"loss": outputs.loss}

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=128,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.2,
        length_penalty=1.0,
        num_captions=1,
        temperature=1.0,
    ):
        rad_occ_mem0 = samples["rad_occ_mem0"]
        text_input = samples.get("prompt", samples["text_input"])
        qa_id = samples["qa_id"]
        if isinstance(text_input, str):
            text_input = [text_input]

        # Use model device as source of truth to avoid stale CPU inputs from dataloader side.
        device = next(self.qwen_model.parameters()).device

        with self.maybe_autocast(dtype=torch.float32):
            prefix_embeds, prefix_atts = self._build_prefix(rad_occ_mem0, text_input, device)

            if self.use_chat_template and hasattr(self.qwen_tokenizer, "apply_chat_template"):
                prompt_ids, prompt_attention_mask = self._tokenize_chat_infer(
                    text_input=text_input,
                    device=device,
                )
                prompt_embeds = self.qwen_model.get_input_embeddings()(prompt_ids)
                if self.use_radar_token and self.radar_token_id is not None:
                    full_embeds, full_atts, _ = self._replace_radar_token_with_prefix(
                        token_ids=prompt_ids,
                        token_embeds=prompt_embeds,
                        token_attention_mask=prompt_attention_mask,
                        prefix_embeds=prefix_embeds,
                        token_labels=None,
                        prepend_prefix_if_missing=True,
                    )
                else:
                    full_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
                    full_atts = torch.cat([prefix_atts, prompt_attention_mask], dim=1)
            else:
                text_input = self._prepend_radar_token_batch_if_missing(text_input)
                input_tokens = self.qwen_tokenizer(
                    text_input,
                    padding="longest",
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(device)

                input_embeds = self.qwen_model.get_input_embeddings()(input_tokens.input_ids)
                if self.use_radar_token and self.radar_token_id is not None:
                    full_embeds, full_atts, _ = self._replace_radar_token_with_prefix(
                        token_ids=input_tokens.input_ids,
                        token_embeds=input_embeds,
                        token_attention_mask=input_tokens.attention_mask,
                        prefix_embeds=prefix_embeds,
                        token_labels=None,
                        prepend_prefix_if_missing=True,
                    )
                else:
                    full_embeds = torch.cat([prefix_embeds, input_embeds], dim=1)
                    full_atts = torch.cat([prefix_atts, input_tokens.attention_mask], dim=1)

            if not use_nucleus_sampling:
                top_p = None

            generated = self.qwen_model.generate(
                inputs_embeds=full_embeds,
                attention_mask=full_atts,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )

        return self.qwen_tokenizer.batch_decode(generated, skip_special_tokens=True)

    def predict_answers(self, samples, **kwargs):
        output_text = self.generate(samples, **kwargs)
        if self._apply_lemmatizer or (
            "apply_lemmatizer" in samples and samples["apply_lemmatizer"]
        ):
            output_text = self._lemmatize(output_text)
        return output_text

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)
            words = [token.lemma_ if token.pos_ in ["NOUN", "VERB"] else token.text for token in doc]
            return " ".join(words)

        return [apply(answer) for answer in answers]
    
    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.amp.autocast('cuda',dtype=dtype)
        else:
            return contextlib.nullcontext()

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError as e:
                raise ImportError(
                    "Install spacy and en_core_web_sm for lemmatization support"
                ) from e
        return self._lemmatizer

    @classmethod
    def from_config(cls, cfg):
        model = cls(
            Z_rad=cfg.get("Z_rad", 200),
            Y_rad=cfg.get("Y_rad", 8),
            X_rad=cfg.get("X_rad", 200),
            latent_dim=cfg.get("latent_dim", 128),
            use_radar_occupancy_map=cfg.get("use_radar_occupancy_map", False),
            use_rpn_radar=cfg.get("use_rpn_radar", False),
            num_query_token=cfg.get("num_query_token", 32),
            llm_model=cfg.get("llm_model", "/home/renault/repo/models/Qwen2.5-3B-Instruct"),
            prompt=cfg.get("prompt", ""),
            max_txt_len=cfg.get("max_txt_len", 128),
            max_output_txt_len=cfg.get("max_output_txt_len", 256),
            apply_lemmatizer=cfg.get("apply_lemmatizer", False),
            qformer_text_input=cfg.get("qformer_text_input", True),
            has_lora=cfg.get("has_lora", True),
            save_radar_bev_debug=cfg.get("save_radar_bev_debug", False),
            radar_bev_debug_dir=cfg.get("radar_bev_debug_dir", "outputs/radar_bev_debug"),
            radar_bev_debug_max_items=cfg.get("radar_bev_debug_max_items", 1),
            use_chat_template=cfg.get("use_chat_template", True),
            include_system_prompt=cfg.get("include_system_prompt", True),
            system_prompt=cfg.get("system_prompt", "You are a helpful assistant."),
            use_radar_token=cfg.get("use_radar_token", True),
            radar_token=cfg.get("radar_token", "<radar>"),
        )

        model.load_checkpoint_from_config(cfg)

        return model
    
    #Adapted from BaseTask to add the loading of the qformer pretrained weights if specified in the config file.
    def load_checkpoint_from_config(self, cfg, **kwargs):
        """
        Load checkpoint as specified in the config file.

        If load_qformer

        If load_finetuned is True, load the finetuned model; otherwise, load the pretrained model.
        When loading the pretrained model, each task-specific architecture may define their
        own load_from_pretrained() method.
        """
        load_finetuned = cfg.get("load_finetuned", True)
        if load_finetuned:
            finetune_path = cfg.get("finetuned", None)
            assert (
                finetune_path is not None
            ), "Found load_finetuned is True, but finetune_path is None."
            self.load_checkpoint(url_or_filename=finetune_path)
        else:
            load_qformer_pretrained = cfg.get("load_qformer_pretrained", False)
            if load_qformer_pretrained:
                qformer_pretrain_path = cfg.get("qformer_pretrained", None)
                assert (
                    qformer_pretrain_path is not None
                ), "Found load_qformer_pretrained is True, but qformer_pretrain_path is None."
                self.load_from_pretrained(url_or_filename=qformer_pretrain_path)
                print_log("Loaded QFormer pretrained weights from %s" % qformer_pretrain_path, logger=self.logger_name)

            load_voxelnet_pretrained = cfg.get("load_voxelnet_pretrained", False)
            if load_voxelnet_pretrained:
                voxelnet_pretrain_path = cfg.get("voxelnet_pretrained", None)
                assert (
                    voxelnet_pretrain_path is not None
                ), "Found load_voxelnet_pretrained is True, but voxelnet_pretrain_path is None."
                #need to do self.load and not self.radar_encoder.load bc the name of the keys. 
                self.load_from_pretrained(url_or_filename=voxelnet_pretrain_path, weights_only=False)
                print_log("Loaded VoxelNet pretrained weights from %s" % voxelnet_pretrain_path, logger=self.logger_name)

            #Maybe need to do and else if to do one of the two
            #It still work if the ckpt from the load_pretrained has loaded the qformer weights.
            load_pretrained = cfg.get("load_pretrained", True)

            if load_pretrained:
                # load pre-trained weights
                pretrain_path = cfg.get("pretrained", None)
                assert "Found load_finetuned is False, but pretrain_path is None."
                self.load_from_pretrained(url_or_filename=pretrain_path, **kwargs)
    
    #get from the blip2.py file from LAVIS repo.
    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel.from_pretrained(
            "bert-base-uncased", config=encoder_config
        )
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens
    
    #get from the blip2.py file from LAVIS repo. 
    @classmethod
    def init_tokenizer(cls, truncation_side="right"):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side=truncation_side)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer

    #get from blip2.py file from LAVIS repo.
    def load_from_pretrained(self, url_or_filename, weights_only=True):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu", weights_only=weights_only)
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu", weights_only=weights_only)
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        if "model_state_dict" in checkpoint:
            checkpoint["model"] = checkpoint.pop("model_state_dict")

        state_dict = checkpoint["model"]

        msg = self.load_state_dict(state_dict, strict=False)

        # logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg