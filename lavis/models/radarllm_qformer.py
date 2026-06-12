"""
Radar-text pretraining model with ITC + ITM + LM losses.
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn import functional as F

from lavis.common.registry import registry
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from lavis.models.radarllm_base import RadarLLMBase


@registry.register_model("radarllm_qformer")
class RadarLLMQformer(RadarLLMBase):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/radarllm_qformer_pretrain.yaml",
    }

    def __init__(
        self,
        Z_rad: int = 200,
        Y_rad: int = 8,
        X_rad: int = 200,
        latent_dim: int = 128,
        use_radar_occupancy_map: bool = False,
        use_rpn_radar: bool = False,
        num_query_token: int = 32,
        max_txt_len: int = 128,
        qformer_text_input: bool = True,
        embed_dim: int = 256,
        itm_hard_negative_topk: int = 4,
        itm_loss_weight: float = 2.0,
        itm_warmup_steps: int = 0,
        debug_radar_diversity: bool = False,
    ):
        super().__init__(
            Z_rad=Z_rad,
            Y_rad=Y_rad,
            X_rad=X_rad,
            latent_dim=latent_dim,
            use_radar_occupancy_map=use_radar_occupancy_map,
            use_rpn_radar=use_rpn_radar,
            num_query_token=num_query_token,
            max_txt_len=max_txt_len,
            qformer_text_input=qformer_text_input,
        )

        self.radar_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)
        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.temp_min = 0.01
        self.temp_max = 0.07  # prevent collapse via temperature inflation
        self.itm_hard_negative_topk = itm_hard_negative_topk
        self.itm_loss_weight = itm_loss_weight
        # Number of global gradient steps over which itm_loss_weight is linearly
        # ramped from 0 to its full value.  Set to 0 to disable warmup.
        self.itm_warmup_steps = itm_warmup_steps
        self.debug_radar_diversity = debug_radar_diversity
        #Select hardcandidate for ITM loss, if True, select hard negative samples from the batch, otherwise select random negative samples
        self.hardcandidate = True

    def _get_rank(self) -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return 0

    def _log_embedding_diversity(self, x: torch.Tensor, tag: str) -> None:
        """Log quick collapse diagnostics for a batch of embeddings [bs, n, d]."""
        if x.ndim != 3:
            print(f"[RADAR-DBG] {tag}: expected 3D tensor [bs, n, d], got shape={tuple(x.shape)}")
            return

        bs = x.size(0)
        x_f = x.float()

        # Basic value stats catch dead activations/saturation quickly.
        print(
            f"[RADAR-DBG] {tag}: shape={tuple(x.shape)}  "
            f"mean={x_f.mean():.4f} std={x_f.std():.4f}  "
            f"min={x_f.min():.4f} max={x_f.max():.4f}"
        )

        if bs < 2:
            print(f"[RADAR-DBG] {tag}: bs<2, skipping inter-sample diversity checks")
            return

        pooled = F.normalize(x_f.mean(dim=1), dim=-1)
        sim = torch.mm(pooled, pooled.t())
        dist_mat = torch.cdist(pooled, pooled, p=2)
        off_mask = ~torch.eye(bs, dtype=torch.bool, device=x.device)

        off_sim = sim[off_mask]
        off_dist = dist_mat[off_mask]
        near_duplicate_ratio = (off_sim > 0.99).float().mean().item()

        print(
            f"[RADAR-DBG] {tag}: inter-sample cosine offdiag "
            f"mean={off_sim.mean():.4f} std={off_sim.std():.4f} "
            f"min={off_sim.min():.4f} max={off_sim.max():.4f}"
        )
        print(
            f"[RADAR-DBG] {tag}: inter-sample L2 offdiag "
            f"mean={off_dist.mean():.4f} std={off_dist.std():.4f} "
            f"min={off_dist.min():.4f} max={off_dist.max():.4f}"
        )
        if near_duplicate_ratio > 0.5:
            print(
                f"[RADAR-DBG] {tag}: WARNING near-duplicate ratio={near_duplicate_ratio:.1%} "
                f"(offdiag cosine>0.99)"
            )

    def _log_sparse_radar_input_diversity(self, rad_occ_mem0, tag: str) -> None:
        """Log batch diversity for sparse radar inputs before the radar encoder."""
        # Expect the sparse radar tuple: per-voxel features, voxel coordinates, and per-sample occupancy counts.
        if not isinstance(rad_occ_mem0, (tuple, list)) or len(rad_occ_mem0) != 3:
            print(f"[RADAR-DBG] {tag}: expected (voxel_features, voxel_coords, num_occupied)")
            return

        voxel_features, voxel_coords, number_of_occupied_voxels = rad_occ_mem0
        # Normalize dtypes for stable statistics and indexing.
        voxel_features = voxel_features.float()
        voxel_coords = voxel_coords.long()
        number_of_occupied_voxels = number_of_occupied_voxels.long()

        # Coordinate format can be either packed (N, 4 with batch id) or padded per-batch (B, M, C).
        if voxel_coords.ndim not in (2, 3):
            print(f"[RADAR-DBG] {tag}: unexpected voxel_coords shape={tuple(voxel_coords.shape)}")
            return

        batch_size = int(number_of_occupied_voxels.numel())
        num_occupied_flat = number_of_occupied_voxels.reshape(-1)

        if voxel_coords.ndim == 3:
            # In padded mode, first dimension must match batch size.
            if voxel_coords.size(0) != batch_size:
                print(
                    f"[RADAR-DBG] {tag}: batch mismatch coords={tuple(voxel_coords.shape)} "
                    f"num_occupied={tuple(number_of_occupied_voxels.shape)}"
                )
                return
        else:
            # In packed mode, first column stores sample index in the batch.
            batch_ids = voxel_coords[:, 0]

        # Reduce point-level structure to one feature vector per voxel.
        if voxel_features.ndim == 4:
            voxel_point_summary = voxel_features.mean(dim=2)
        elif voxel_features.ndim == 3:
            voxel_point_summary = voxel_features
        elif voxel_features.ndim == 2:
            voxel_point_summary = voxel_features
        else:
            print(f"[RADAR-DBG] {tag}: unexpected voxel_features shape={tuple(voxel_features.shape)}")
            return

        per_sample_summary = []
        per_sample_counts = []
        coord_summaries = []
        empty_samples = []

        # Build one pooled feature vector and one pooled coordinate vector per sample.
        for batch_idx in range(batch_size):
            occupied_count = int(num_occupied_flat[batch_idx].item())
            per_sample_counts.append(occupied_count)

            if occupied_count == 0:
                empty_samples.append(batch_idx)
                # Keep tensor shapes consistent for downstream stack operations.
                per_sample_summary.append(torch.zeros(voxel_point_summary.size(-1), device=voxel_point_summary.device))
                coord_dim = voxel_coords.size(-1) - (1 if voxel_coords.ndim == 2 else 0)
                coord_summaries.append(torch.zeros(coord_dim, device=voxel_features.device))
                continue

            if voxel_coords.ndim == 3:
                # Padded mode: use the first occupied_count entries for this sample.
                sample_features = voxel_point_summary[batch_idx, :occupied_count]
                sample_coords = voxel_coords[batch_idx, :occupied_count].float()
            else:
                # Packed mode: filter rows by batch id.
                sample_mask = batch_ids == batch_idx
                sample_features = voxel_point_summary[sample_mask]
                sample_coords = voxel_coords[sample_mask, 1:].float()

            per_sample_summary.append(sample_features.mean(dim=0))
            coord_summaries.append(sample_coords.mean(dim=0))

        feature_summary = torch.stack(per_sample_summary, dim=0)
        coord_summary = torch.stack(coord_summaries, dim=0)

        print(
            f"[RADAR-DBG] {tag}: voxel_features shape={tuple(voxel_features.shape)}  "
            f"voxel_coords shape={tuple(voxel_coords.shape)}  occupied={per_sample_counts}"
        )
        print(
            f"[RADAR-DBG] {tag}: feature mean={feature_summary.mean():.4f} std={feature_summary.std():.4f}  "
            f"coord mean={coord_summary.mean():.4f} std={coord_summary.std():.4f}"
        )

        if batch_size < 2:
            return

        # Compare pooled sample descriptors to detect near-duplicate radar inputs.
        norm_feature_summary = F.normalize(feature_summary, dim=-1)
        feature_sim = torch.mm(norm_feature_summary, norm_feature_summary.t())
        feature_dist = torch.cdist(norm_feature_summary, norm_feature_summary, p=2)

        # Also compare mean coordinates as a geometric sanity check.
        norm_coord_summary = F.normalize(coord_summary, dim=-1)
        coord_sim = torch.mm(norm_coord_summary, norm_coord_summary.t())

        off_mask = ~torch.eye(batch_size, dtype=torch.bool, device=feature_summary.device)
        feature_off_sim = feature_sim[off_mask]
        feature_off_dist = feature_dist[off_mask]
        coord_off_sim = coord_sim[off_mask]

        print(
            f"[RADAR-DBG] {tag}: pooled feature cosine offdiag "
            f"mean={feature_off_sim.mean():.4f} std={feature_off_sim.std():.4f} "
            f"min={feature_off_sim.min():.4f} max={feature_off_sim.max():.4f}"
        )
        print(
            f"[RADAR-DBG] {tag}: pooled feature L2 offdiag "
            f"mean={feature_off_dist.mean():.4f} std={feature_off_dist.std():.4f} "
            f"min={feature_off_dist.min():.4f} max={feature_off_dist.max():.4f}"
        )
        print(
            f"[RADAR-DBG] {tag}: mean coord cosine offdiag "
            f"mean={coord_off_sim.mean():.4f} std={coord_off_sim.std():.4f} "
            f"min={coord_off_sim.min():.4f} max={coord_off_sim.max():.4f}"
        )

        near_duplicate_ratio = (feature_off_sim > 0.99).float().mean().item()
        if near_duplicate_ratio > 0.5:
            print(
                f"[RADAR-DBG] {tag}: WARNING near-duplicate pooled input ratio={near_duplicate_ratio:.1%} "
                f"(offdiag cosine>0.99)"
            )

        if empty_samples:
            print(f"[RADAR-DBG] {tag}: WARNING empty radar samples at batch indices={empty_samples}")

    def _encode_text(self, text_input, device: torch.device):
        text_tokens = self.tokenizer(
            text_input,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(device)

        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)
        return text_tokens, text_output, text_feat

    def _log_voxel_iou(self, voxel_coords, num_occupied_flat, batch_size: int, tag: str) -> None:
        """Log pairwise IoU of occupied voxel sets between batch samples."""
        # Build a set of (z, y, x) tuples per sample using Python sets for simplicity.
        # voxel_coords is either (B, max_pts, 3) or (N, 4) with batch_id in col-0.
        voxel_sets = []
        for b in range(batch_size):
            n = int(num_occupied_flat[b].item())
            if voxel_coords.ndim == 3:
                coords = voxel_coords[b, :n].tolist()
            else:
                mask = voxel_coords[:, 0] == b
                coords = voxel_coords[mask, 1:].tolist()
            voxel_sets.append(set(tuple(c) for c in coords))

        iou_vals = []
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                inter = len(voxel_sets[i] & voxel_sets[j])
                union = len(voxel_sets[i] | voxel_sets[j])
                iou = inter / union if union > 0 else 0.0
                iou_vals.append(iou)

        if not iou_vals:
            return
        mean_iou = sum(iou_vals) / len(iou_vals)
        max_iou  = max(iou_vals)
        min_iou  = min(iou_vals)
        print(
            f"[RADAR-DBG] {tag}: voxel IoU offdiag "
            f"mean={mean_iou:.4f} min={min_iou:.4f} max={max_iou:.4f} "
            f"pairs={iou_vals}  "
            f"(0=no overlap; 1=identical voxel sets)"
        )
        if max_iou > 0.8:
            print(f"  WARNING: some sample pairs share >80% of occupied voxels (max IoU={max_iou:.3f})")

    def _log_bev_histograms(self, voxel_coords, num_occupied_flat, batch_size: int,
                             bev_bins: int, tag: str) -> None:
        """Log BEV (Z, X) occupancy histograms per sample and their pairwise similarities."""
        histograms = []
        for b in range(batch_size):
            n = int(num_occupied_flat[b].item())
            if n == 0:
                histograms.append(torch.zeros(bev_bins * bev_bins, device=voxel_coords.device))
                continue
            if voxel_coords.ndim == 3:
                # coords shape (n, 3) — columns are (z, y, x) or similar; use col 0 and 2 for BEV
                z = voxel_coords[b, :n, 0].float()
                x = voxel_coords[b, :n, 2].float() if voxel_coords.size(2) >= 3 else voxel_coords[b, :n, 1].float()
            else:
                mask = voxel_coords[:, 0] == b
                sub = voxel_coords[mask, 1:].float()
                z = sub[:, 0]
                x = sub[:, 2] if sub.size(1) >= 3 else sub[:, 1]

            z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)
            x_norm = (x - x.min()) / (x.max() - x.min() + 1e-6)
            z_idx  = (z_norm * (bev_bins - 1)).long().clamp(0, bev_bins - 1)
            x_idx  = (x_norm * (bev_bins - 1)).long().clamp(0, bev_bins - 1)
            flat   = z_idx * bev_bins + x_idx
            hist   = torch.zeros(bev_bins * bev_bins, device=voxel_coords.device)
            hist.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float))
            histograms.append(hist)

        hists = torch.stack(histograms, dim=0)              # (B, bev_bins²)
        hists_norm = F.normalize(hists + 1e-8, dim=-1)      # L2-normalise for cosine
        sim = torch.mm(hists_norm, hists_norm.t())
        off_mask = ~torch.eye(batch_size, dtype=torch.bool, device=sim.device)
        off_sim = sim[off_mask]
        print(
            f"[RADAR-DBG] {tag}: BEV histogram cosine offdiag (bins={bev_bins}×{bev_bins}) "
            f"mean={off_sim.mean():.4f} std={off_sim.std():.4f} "
            f"min={off_sim.min():.4f} max={off_sim.max():.4f}  "
            f"(1=identical BEV distribution; 0=no overlap)"
        )
        if off_sim.mean().item() > 0.9:
            print(f"  WARNING: BEV distributions are very similar across samples (mean cos={off_sim.mean():.3f})")

    def _build_bev_atts(self, rad_occ_mem0, device: torch.device) -> torch.Tensor:
        """
        Build a BEV attention mask of shape (B, Z * X) for the QFormer cross-attention.

        A position is set to 1 (attend) if at least one radar voxel occupies that
        (z, x) column, and 0 (ignore) otherwise.  This focuses the QFormer query
        tokens exclusively on the BEV positions that carry actual radar
        signal, preventing the constant BatchNorm background
        from drowning the occupied-voxel features during cross-attention.

        Coordinate convention matches VoxelNet's voxel_indexing:
            voxel_coords[b, i] = (z, y, x)  — Y is the height axis, collapsed in BEV.
        The flat BEV index is therefore: z * X + x.

        Falls back to all-ones for any sample with zero occupied voxels so the
        QFormer still receives a valid (non-empty) key sequence.
        """
        _, voxel_coords, num_occupied = rad_occ_mem0
        B = voxel_coords.size(0)
        # Maximum voxel slots allocated in the padded tensor.
        N = voxel_coords.size(1)

        bev_z = self.Z_rad   # 200 by default
        bev_x = self.X_rad   # 200 by default
        seq_len = bev_z * bev_x  # 40,000 for default 200 × 200 BEV

        atts = torch.zeros(B, seq_len, dtype=torch.long, device=device)
        num_occupied_flat = num_occupied.reshape(-1).long()
        # Move coords to target device once to avoid repeated transfers.
        voxel_coords_dev = voxel_coords.long().to(device)

        for b in range(B):
            n = int(num_occupied_flat[b].item())
            # Clamp to the number of allocated padded slots.
            n = min(n, N)

            if n == 0:
                # No radar points for this sample: fall back to full attention so
                # the QFormer is not handed an all-zero key sequence.
                atts[b].fill_(1)
                continue

            # Extract (z, x) coordinates for the n valid voxels and clamp to grid.
            z_idx = voxel_coords_dev[b, :n, 0].clamp(0, bev_z - 1)
            x_idx = voxel_coords_dev[b, :n, 2].clamp(0, bev_x - 1)

            # Collapse the Y (height) dimension into a 2-D BEV flat index.
            flat_idx = z_idx * bev_x + x_idx
            # scatter_ sets every listed position to 1 (duplicates are harmless).
            atts[b].scatter_(0, flat_idx, 1)

        return atts

    def _encode_radar_queries(self, rad_occ_mem0, device: torch.device):
        rank = self._get_rank()
        do_debug = rank == 0 and self.debug_radar_diversity

        if do_debug:
            with torch.no_grad():
                self._log_sparse_radar_input_diversity(rad_occ_mem0, "rad_occ_mem0")
                # --- extra diagnostics: voxel IoU + BEV histograms ---
                if isinstance(rad_occ_mem0, (tuple, list)) and len(rad_occ_mem0) == 3:
                    _vf, _vc, _nv = rad_occ_mem0
                    _nv_flat = _nv.reshape(-1).long()
                    _bs = int(_nv_flat.numel())
                    _vc_cpu = _vc.long().cpu()
                    _nv_cpu = _nv_flat.cpu()
                    if _bs >= 2:
                        self._log_voxel_iou(_vc_cpu, _nv_cpu, _bs, "rad_occ_mem0")
                        self._log_bev_histograms(_vc_cpu, _nv_cpu, _bs,
                                                 bev_bins=20, tag="rad_occ_mem0")

        rad_tokens = self._encode_radar(rad_occ_mem0, device)

        # Build a sparse attention mask: only attend to BEV positions that correspond
        # to at least one occupied radar voxel.  Attending to all 40,000 positions
        # would dilute the ~150 occupied-voxel signals by the ~39,850 uniform
        # background tokens produced by BatchNorm on zero-input voxels.
        rad_atts = self._build_bev_atts(rad_occ_mem0, device)

        if do_debug:
            with torch.no_grad():
                # Log the fraction of attended BEV positions to verify sparsity.
                occ_fraction = rad_atts.float().mean().item()
                print(
                    f"[RADAR-DBG] rad_atts: attended positions per sample = "
                    f"{rad_atts.sum(dim=1).tolist()} / {rad_atts.size(1)} "
                    f"(mean occupancy fraction = {occ_fraction:.4f})"
                )
                self._log_embedding_diversity(rad_tokens, "pre-qformer rad_tokens")

        query_tokens = self.query_tokens.expand(rad_tokens.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=rad_tokens,
            encoder_attention_mask=rad_atts,
            use_cache=True,
            return_dict=True,
        )

        radar_proj_out = self.radar_proj(query_output.last_hidden_state)
        radar_feats = F.normalize(radar_proj_out, dim=-1)

        if do_debug:
            with torch.no_grad():
                self._log_embedding_diversity(query_output.last_hidden_state, "post-qformer hidden")
                self._log_embedding_diversity(radar_proj_out, "post-proj pre-norm")
                self._log_embedding_diversity(radar_feats, "post-proj normalized")

        # Return the sparse mask alongside the tokens so callers can reuse
        # it for any subsequent QFormer forward pass (e.g. ITM).
        return query_output, rad_tokens, radar_feats, rad_atts

    def _contrastive_loss(self, radar_feats, text_feat, samples):
        'Compute a constrastive loss between the radar features and the text features computed with two differents qformer'
        'Adapted from BLIP2_qformer, clamp the temperature to prevent collapse via inflation'
        # Clamp temperature to prevent collapse via inflation
        self.temp.data.clamp_(self.temp_min, self.temp_max)
        rank = self._get_rank()
        do_debug = rank == 0 and self.debug_radar_diversity

        if do_debug:
            with torch.no_grad():
                # --- feature diversity: if mean_pairwise_cosine → 1.0, collapse is occurring ---
                rad_mean = F.normalize(radar_feats.mean(dim=1), dim=-1)  # [bs, embed_dim]
                rad_self_sim = torch.mm(rad_mean, rad_mean.t())
                bs_local = rad_mean.size(0)
                off_mask = ~torch.eye(bs_local, dtype=torch.bool, device=rad_self_sim.device)
                rad_off = rad_self_sim[off_mask]
                txt_self_sim = torch.mm(text_feat, text_feat.t())
                txt_off = txt_self_sim[off_mask]
                print(
                    f"[ITC] temp={self.temp.item():.5f}  "
                    f"radar pairwise cosine mean={rad_off.mean():.4f} std={rad_off.std():.4f}  "
                    f"text pairwise cosine mean={txt_off.mean():.4f} std={txt_off.std():.4f}  "
                    f"(collapse if mean→1; diverse if mean≈0±0.1)"
                )

        radar_feats_all = concat_all_gather(radar_feats) # [batch_size*num_gpu, num_query_tokens, embed_dim]
        text_feat_all = concat_all_gather(text_feat) # [batch_size*num_gpu, embed_dim]

        # radar query-text similarity: aggregate across all query tokens
        #sim_q2t = torch.einsum("bqd,kd->bkq", radar_feats, text_feat_all)# [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_q2t = torch.matmul(
            radar_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
        ).squeeze()
        sim_r2t, _ = sim_q2t.max(-1) # take max-similarity between all query tokens.
        sim_r2t = sim_r2t / self.temp

        # text- radar query similarity: aggregate across all query tokens
        #sim_t2q = torch.einsum("bd,kqd->bkq", text_feat, radar_feats_all) #[batch_size, batch_size*num_gpu, num_query_tokens]
        sim_t2q = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1), radar_feats_all.permute(0, 2, 1)
        ).squeeze()
        sim_t2r, _ = sim_t2q.max(-1) # take max-similarity between all query tokens.
        sim_t2r = sim_t2r / self.temp

        bs = radar_feats.size(0)
        rank = self._get_rank()
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(radar_feats.device)

        #keep the image_id and not qa_id bc some qa pairs share the same radar input ( which is annotated by image_id )
        if "image_id" in samples:
            image_ids = samples["image_id"].view(-1, 1)
            image_ids_all = concat_all_gather(image_ids)
            pos_idx = torch.eq(image_ids, image_ids_all.t()).float() #build a [batch_size, batch_size*num_gpu] matrix where (i,j) for positive pair is 1 and negative pair is 0
            sim_targets = pos_idx / pos_idx.sum(1, keepdim=True).clamp_min(1.0) #normalize the target distribution so that each positive pair gets equal weight and sum of each row is 1
            sim_targets = 0.9 * sim_targets + 0.1 * torch.ones_like(sim_targets) / sim_targets.size(1) #label smoothing

            loss_t2r = -torch.sum(F.log_softmax(sim_t2r, dim=1) * sim_targets, dim=1).mean()
            loss_r2t = -torch.sum(F.log_softmax(sim_r2t, dim=1) * sim_targets, dim=1).mean()
            loss_itc = (loss_t2r + loss_r2t) / 2
        else:
            loss_itc = (
                F.cross_entropy(sim_r2t, targets, label_smoothing=0.1)
                + F.cross_entropy(sim_t2r, targets, label_smoothing=0.1)
            ) / 2

        return loss_itc, sim_r2t, sim_t2r

    def _matching_loss(self, rad_tokens, rad_atts, text_tokens, sim_r2t, sim_t2r, device: torch.device, samples):
        """Compute the ITM loss using hard negatives selected based on the ITC similarity scores."""
        "Adapted from blip2_qformer.py with the choice of hardnegative samples, "
        #Bring the radar tokens contiguous in memory
        rad_tokens = rad_tokens.contiguous()
        text_input_ids_world = concat_all_gather(text_tokens.input_ids)
        text_attention_mask_world = concat_all_gather(text_tokens.attention_mask)
        rad_tokens_world = all_gather_with_grad(rad_tokens)
        # Gather the sparse occupancy masks from all GPUs so that, when we pick
        # a hard-negative radar token from another GPU, we can reuse its mask.
        rad_atts_world = concat_all_gather(rad_atts)  # (total_B, seq_len)

        bs = rad_tokens.size(0)
        rank = self._get_rank()
        do_debug = rank == 0 and self.debug_radar_diversity

        with torch.no_grad():
            # ---- log raw cosine similarities BEFORE masking so -10000 doesn't pollute stats ----
            if do_debug:
                temp_val = self.temp.item()
                # sim_r2t is already /temp; multiply back to get cosine similarities
                raw_cosine = sim_r2t * temp_val
                print(
                    f"\n[ITM] temp={temp_val:.5f}  "
                    f"raw cosine sim_r2t (before mask)  "
                    f"mean={raw_cosine.mean():.4f}  std={raw_cosine.std():.4f}  "
                    f"min={raw_cosine.min():.4f}  max={raw_cosine.max():.4f}  "
                    f"(cosine in [-1,1]; discriminative if std>0.05)"
                )
                # pos diagonal: how similar are the actual matched pairs?
                pos_cosine = raw_cosine[torch.arange(bs), rank * bs + torch.arange(bs)]
                # mean of off-diag
                off_mask = torch.ones_like(raw_cosine, dtype=torch.bool)
                off_mask[torch.arange(bs), rank * bs + torch.arange(bs)] = False
                neg_cosine = raw_cosine[off_mask]
                print(
                    f"[ITM] cosine pos(matched)={pos_cosine.mean():.4f}  "
                    f"neg(unmatched)={neg_cosine.mean():.4f}  "
                    f"margin={pos_cosine.mean() - neg_cosine.mean():.4f}  "
                    f"(ideal: pos>>neg; margin>0.1 means ITC has learned something)"
                )

            if "image_id" in samples:
                image_ids = samples["image_id"].view(-1, 1)
                image_ids_all = concat_all_gather(image_ids)
                mask = torch.eq(image_ids, image_ids_all.t())
                sim_t2r = sim_t2r.clone()
                sim_r2t = sim_r2t.clone()
                sim_t2r.masked_fill_(mask, -10000)
                sim_r2t.masked_fill_(mask, -10000)
            else:
                sim_t2r = sim_t2r.clone()
                sim_r2t = sim_r2t.clone()
                sim_t2r[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)
                sim_r2t[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)

            weights_t2r = F.softmax(sim_t2r, dim=1)
            weights_r2t = F.softmax(sim_r2t, dim=1)

            assert not torch.isnan(weights_t2r).any(), "NaN in weights_t2r"
            assert not torch.isnan(weights_r2t).any(), "NaN in weights_r2t"

            if do_debug:
                # ---- weights distribution: are hard negatives well-peaked or flat? ----
                max_w_t2r, argmax_w_t2r = weights_t2r.max(dim=1)
                max_w_r2t, argmax_w_r2t = weights_r2t.max(dim=1)
                pos_w_t2r = weights_t2r[torch.arange(bs), rank * bs + torch.arange(bs)]  # weight on the (masked) positive
                pos_w_r2t = weights_r2t[torch.arange(bs), rank * bs + torch.arange(bs)]
                print(
                    f"\n[ITM] weights_t2r  max={max_w_t2r.mean():.4f}  argmax={argmax_w_t2r.tolist()}  pos_weight={pos_w_t2r.mean():.6f} (should be ~0)"
                )
                print(
                    f"[ITM] weights_r2t  max={max_w_r2t.mean():.4f}  argmax={argmax_w_r2t.tolist()}  pos_weight={pos_w_r2t.mean():.6f} (should be ~0)"
                )
                # uniform weight would be 1/total_bs — flag if no concentration
                total_bs = bs * max(1, weights_t2r.size(1) // bs)
                if max_w_t2r.mean().item() < 2.0 / total_bs:
                    print(f"  WARNING: weights_t2r is near-uniform (max≈{max_w_t2r.mean():.4f}, uniform={1/total_bs:.4f}) — hard-negatives may be random")

        # Select hard negatives from top-k highest-similarity candidates.
        hard_k = min(self.itm_hard_negative_topk, sim_t2r.size(1))
        assert hard_k > 0, "No candidates available for hard negative mining"

        # select a hard negative radar token for each text
        rad_tokens_neg = []
        rad_neg_indices = []
        for b in range(bs):
            if self.hardcandidate:
                hard_candidates = torch.topk(sim_t2r[b], k=hard_k, dim=0).indices
                pos_idx = rank * bs + b
                hard_candidates = hard_candidates[hard_candidates != pos_idx]
                if hard_candidates.numel() == 0:
                    all_indices = torch.arange(sim_t2r.size(1), device=sim_t2r.device)
                    hard_candidates = all_indices[all_indices != pos_idx]
                pick = torch.randint(hard_candidates.numel(), (1,), device=hard_candidates.device).item()
                neg_idx = hard_candidates[pick].item()
            else: 
                neg_idx = torch.multinomial(weights_t2r[b], 1).item()
            rad_neg_indices.append(neg_idx)
            rad_tokens_neg.append(rad_tokens_world[neg_idx])
        rad_tokens_neg = torch.stack(rad_tokens_neg, dim=0)
        # Collect the occupancy mask for each selected negative radar token,
        # using the same index that was used to pick the radar token itself.
        rad_atts_neg = torch.stack([rad_atts_world[i] for i in rad_neg_indices], dim=0)

        # select a hard negative text for each radar token
        text_ids_neg = []
        text_atts_neg = []
        text_neg_indices = []
        for b in range(bs):
            if self.hardcandidate:
                hard_candidates = torch.topk(sim_r2t[b], k=hard_k, dim=0).indices
                pos_idx = rank * bs + b
                hard_candidates = hard_candidates[hard_candidates != pos_idx]
                if hard_candidates.numel() == 0:
                    all_indices = torch.arange(sim_r2t.size(1), device=sim_r2t.device)
                    hard_candidates = all_indices[all_indices != pos_idx]
                pick = torch.randint(hard_candidates.numel(), (1,), device=hard_candidates.device).item()
                neg_idx = hard_candidates[pick].item()
            else:
                neg_idx = torch.multinomial(weights_r2t[b], 1).item()
            text_neg_indices.append(neg_idx)
            text_ids_neg.append(text_input_ids_world[neg_idx])
            text_atts_neg.append(text_attention_mask_world[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        if do_debug:
            # ---- are the selected negative tokens actually different from their positives? ----
            print(f"[ITM] hard negative top-k={hard_k}")
            print(f"[ITM] neg radar  indices: {rad_neg_indices}   pos indices: {list(range(rank*bs, rank*bs+bs))}")
            print(f"[ITM] neg text   indices: {text_neg_indices}  pos indices: {list(range(rank*bs, rank*bs+bs))}")
            with torch.no_grad():
                for b in range(bs):
                    rad_same = torch.allclose(rad_tokens_neg[b].float(), rad_tokens[b].float(), atol=1e-5)
                    if rad_same:
                        print(f"  WARNING: rad_tokens_neg[{b}] is identical to rad_tokens[{b}] (neg_idx={rad_neg_indices[b]})")
                # token-id overlap for texts
                for b in range(bs):
                    overlap = (text_ids_neg[b] == text_tokens.input_ids[b]).float().mean().item()
                    if overlap > 0.95:
                        print(f"  WARNING: text_ids_neg[{b}] has {overlap:.1%} token overlap with its positive — near-duplicate negative")

        text_ids_all = torch.cat([text_tokens.input_ids, text_tokens.input_ids, text_ids_neg], dim=0)
        text_atts_all = torch.cat([text_tokens.attention_mask, text_tokens.attention_mask, text_atts_neg], dim=0)

        query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
        query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(device)
        attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

        rad_tokens_all = torch.cat([rad_tokens, rad_tokens_neg, rad_tokens], dim=0)
        # Build the attention mask that mirrors the token order:
        #   [positive radar | negative radar | positive radar]
        # Each slice gets its own sparse occupancy mask so ITM cross-attention
        # attends only to the occupied BEV positions, just like the ITC path.
        rad_atts_all = torch.cat([rad_atts, rad_atts_neg, rad_atts], dim=0)

        output_itm = self.Qformer.bert(
            text_ids_all,
            query_embeds=query_tokens_itm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=rad_tokens_all,
            encoder_attention_mask=rad_atts_all,
            return_dict=True,
        )

        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :]
        logits = self.itm_head(vl_embeddings).mean(dim=1)

        itm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(device)
        loss_itm = F.cross_entropy(logits, itm_labels)

        if do_debug:
            with torch.no_grad():
                probs = F.softmax(logits.detach(), dim=-1)  # [3*bs, 2]
                preds = logits.detach().argmax(dim=-1)      # [3*bs]
                pos_acc  = (preds[:bs] == 1).float().mean().item()
                neg_acc  = (preds[bs:] == 0).float().mean().item()
                # mean prob of being classified as "match" (class 1) per group
                p_match_pos = probs[:bs, 1].mean().item()
                p_match_neg = probs[bs:, 1].mean().item()
                print(
                    f"[ITM] logits  class-0 mean={logits[:, 0].detach().mean():.3f}  class-1 mean={logits[:, 1].detach().mean():.3f}"
                )
                print(
                    f"[ITM] p(match) pos={p_match_pos:.3f}  neg={p_match_neg:.3f}  "
                    f"(random prior: pos=0.333, neg=0.333 — ideal: pos≈1, neg≈0)"
                )
                print(
                    f"[ITM] accuracy  pos={pos_acc:.3f}  neg={neg_acc:.3f}  "
                    f"(random: 0.333 / 0.667 — ideal: 1.0 / 1.0)"
                )
                # Flag if model is just predicting the majority class
                if neg_acc > 0.95 and pos_acc < 0.1:
                    print("  WARNING: model always predicts 'no match' — ITM head collapsed to majority class")
                elif abs(p_match_pos - p_match_neg) < 0.05:
                    print("  WARNING: p(match) is identical for pos and neg — ITM head not discriminating at all")

        # Compute accuracy for positives and negatives separately to get a clearer picture of model performance.
        with torch.no_grad():
            preds = logits.detach().argmax(dim=-1)
            # Positive pairs are the first bs samples (label=1).
            itm_pos_acc = (preds[:bs] == 1).float().mean()
            # Negative pairs are the remaining 2*bs samples (label=0).
            itm_neg_acc = (preds[bs:] == 0).float().mean()

        return loss_itm, itm_pos_acc, itm_neg_acc

    def _lm_loss(self, query_output, text_output, device: torch.device):
        'Train the qformer to generate text based on past query output and past text token'
        'Exactly the same as BLIP2_qformer'
        text = [t + "\n" for t in text_output]
        output_tokens = self.tokenizer(
            text,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(device)

        decoder_input_ids = output_tokens.input_ids.clone() 
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id # replace the first token with [DEC] token to indicate the start of decoding
        labels = decoder_input_ids.masked_fill(decoder_input_ids == self.tokenizer.pad_token_id, -100) #build label with padding

        query_tokens = self.query_tokens.expand(output_tokens.input_ids.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(device)
        attention_mask = torch.cat([query_atts, output_tokens.attention_mask], dim=1)

        #Learn to predict the next text token given the past query output ( radar ) with the last text token, the next text token are masked
        #Loss computed between the label ie the next token and the one predicted.
        lm_output = self.Qformer(
            decoder_input_ids,
            attention_mask=attention_mask,
            past_key_values=query_output.past_key_values,
            return_dict=True,
            labels=labels,
        )
        return lm_output.loss

    def _get_itm_weight(self, samples) -> float:
        """
        Return the effective ITM loss weight for the current training step.

        When itm_warmup_steps > 0 the weight is linearly ramped from 0 to
        self.itm_loss_weight over the first itm_warmup_steps global gradient
        steps.  This prevents ITM from receiving gradient before ITC has built
        a meaningful similarity space, which would otherwise cause the ITM head
        to collapse to the majority class immediately.

        The global step is approximated from the epoch / iteration counters
        injected into every sample dict by the runner.
        """
        # If no warmup, return the full weight.
        if self.itm_warmup_steps <= 0:
            return self.itm_loss_weight

        epoch = samples.get("epoch", 0)
        iters = samples.get("iters", 0)
        num_iters_per_epoch = samples.get("num_iters_per_epoch", 1)
        # Global step counts completed gradient updates, not raw iterations.
        global_step = epoch * num_iters_per_epoch + iters

        # Linear ramp: 0 at step 0, full weight at itm_warmup_steps.
        ramp = min(1.0, global_step / self.itm_warmup_steps)
        return ramp * self.itm_loss_weight

    def forward(self, samples):
        rad_occ_mem0 = samples["rad_occ_mem0"]
        text_input = samples["text_input"]
        text_output = samples.get("text_output", samples["text_input"])

        if isinstance(text_input, str):
            text_input = [text_input]
        if isinstance(text_output, str):
            text_output = [text_output]

        device = self.query_tokens.device
        with self.maybe_autocast(dtype=torch.float32):
            query_output, rad_tokens, radar_feats, rad_atts = self._encode_radar_queries(rad_occ_mem0, device)
            text_tokens, _, text_feat = self._encode_text(text_output, device)

            #ITC: contrastive loss between radar queries and text queries
            loss_itc, sim_r2t, sim_t2r = self._contrastive_loss(radar_feats, text_feat, samples)
            #ITM: matching loss to align radar and text features, which also serves as hard negative mining for ITC
            # Pass the sparse BEV attention mask so ITM attends only to occupied positions.
            loss_itm, itm_pos_acc, itm_neg_acc = self._matching_loss(
                rad_tokens, rad_atts, text_tokens, sim_r2t, sim_t2r, device, samples
            )
            #LM: language modeling loss for text generation conditioned on radar input
            loss_lm = self._lm_loss(query_output, text_output, device)

            # Scale ITM by the current warmup ramp (0 → full weight over itm_warmup_steps).
            itm_weight = self._get_itm_weight(samples)
            loss = loss_itc + itm_weight * loss_itm + loss_lm

        return {
            "loss": loss,
            "loss_itc": loss_itc,
            "loss_itm": loss_itm,
            "loss_lm": loss_lm,
            # ITM accuracy: fraction of positives correctly identified as 'match' (ideal=1.0)
            # and fraction of negatives correctly rejected (ideal=1.0). Random baselines: 0.33 / 0.67.
            "acc_itm_pos": itm_pos_acc,
            "acc_itm_neg": itm_neg_acc,
        }

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
            max_txt_len=cfg.get("max_txt_len", 128),
            qformer_text_input=cfg.get("qformer_text_input", True),
            embed_dim=cfg.get("embed_dim", 256),
            itm_hard_negative_topk=cfg.get("itm_hard_negative_topk", 4),
            itm_loss_weight=cfg.get("itm_loss_weight", 2.0),
            itm_warmup_steps=cfg.get("itm_warmup_steps", 0),
            debug_radar_diversity=cfg.get("debug_radar_diversity", True),
        )

        model.load_checkpoint_from_config(cfg)
        return model
