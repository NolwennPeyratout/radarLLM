import logging

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate

from lavis.datasets.datasets.base_dataset import BaseDataset


_logger = logging.getLogger(__name__)


class NuscenesRadarLLMDataset(BaseDataset):
    """
    Wraps a VizData instance from nuscenes_data.py and converts its tuple
    output into the BLIP2-compatible sample dict expected by RadarLLM:
        {
            "rad_occ_mem0": tuple(voxel_feats, voxel_coords, n_voxels)  # VoxelNet sparse
                         OR torch.Tensor (C, Z, Y, X)                   # dense fallback
            "text_input":  str   — question from LiDAR-LLM-Nu-Caption
            "text_output": str   — answer from LiDAR-LLM-Nu-Caption
            "image_id":    int   — sample index (used for evaluation metrics)
        }

    The dataset is built with seqlen=1 so the time dimension is always 1 and
    gets squeezed away here before returning.
    """

    def __init__(self, viz_data):
        """
        Args:
            viz_data: An instantiated VizData object from nuscenes_data.py.
                      Must have been created with seqlen=1.
        """
        # Skip BaseDataset.__init__ — it requires COCO-style annotation paths.
        # We manage iteration entirely through viz_data.
        self.inner_dataset = viz_data

        # Build a flat index: one dataset item per QA pair.
        # Each entry is (inner_dataset_index, qa_index_within_that_sample).
        self.qa_index = self._build_qa_index()

        _logger.info(
            "NuscenesRadarLLMDataset: %d base samples -> %d QA samples",
            len(self.inner_dataset),
            len(self.qa_index),
        )

    def _resolve_rec_from_inner_index(self, inner_index):
        """
        Resolve the nuScenes sample record used by inner_dataset[inner_index]
        without running heavy tensor loading.

        VizData stores per-item references in self.indices and raw sample records
        in self.ixes. For seqlen=1, indices[inner_index] contains exactly one
        sample index.
        """
        sample_ref = self.inner_dataset.indices[inner_index]

        # For seqlen=1, this is typically array([idx]) or [idx].
        if isinstance(sample_ref, (list, tuple, np.ndarray)):
            if len(sample_ref) == 0:
                return None
            sample_idx = int(sample_ref[0])
        else:
            sample_idx = int(sample_ref)

        if sample_idx < 0 or sample_idx >= len(self.inner_dataset.ixes):
            return None

        return self.inner_dataset.ixes[sample_idx]

    def _build_qa_index(self):
        qa_index = []

        for inner_idx in range(len(self.inner_dataset)):
            rec = self._resolve_rec_from_inner_index(inner_idx)
            if rec is None:
                continue

            questions, answers, qa_ids = self.inner_dataset.getQA(rec)
            
            for qa_id in qa_ids:
                qa_index.append((inner_idx, qa_id))

        return qa_index
    
    def image_id_to_sample_token(self, image_id):
        """Convert image_id → samples_token"""
        rec = self._resolve_rec_from_inner_index(image_id)
        if rec is None:
            return None
        return rec['token']

    def sample_token_to_image_id(self, sample_token):
        """Convert samples_token → image_id"""
        for inner_idx in range(len(self.inner_dataset)):
            rec = self._resolve_rec_from_inner_index(inner_idx)
            if rec is not None and rec['token'] == sample_token:
                return inner_idx
        return -1

    # ------------------------------------------------------------------
    # Required Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.qa_index)

    def __getitem__(self, index):
        inner_idx, qa_id = self.qa_index[index]
        #function to get directly the question, answer for the precise qa_idx
        item = self.inner_dataset.get_single_item_by_qa_index(inner_idx, qa_id)

        # VizData already contains radar preprocessing.
        # We only adapt its tuple output to RadarLLM input dict.
        if self.inner_dataset.radar_encoder_type == "voxel_net":
            (
                _imgs,
                _rots,
                _trans,
                _intrins,
                voxel_input_feature_buffer,
                voxel_coordinate_buffer,
                number_of_occupied_voxels,
                question,
                answer,
                qa_id,
            ) = item

            rad_occ_mem0 = (
                voxel_input_feature_buffer,
                voxel_coordinate_buffer,
                number_of_occupied_voxels,
            )

        else:
            (
                _imgs,
                _rots,
                _trans,
                _intrins,
                question,
                answer,
                qa_id,
            ) = item

            # Non-voxel path is currently unsupported by get_single_item_by_qa_index.
            raise NotImplementedError("get_single_item_by_qa_index currently supports voxel_net only")

        return {
            "rad_occ_mem0": rad_occ_mem0,
            "text_input":   question,
            "text_output":  answer,
            "qa_id":       qa_id,
            "image_id":     inner_idx,
        }

    # ------------------------------------------------------------------
    # LAVIS DataLoader collater
    # ------------------------------------------------------------------

    def collater(self, samples):
        """
        Custom collater to handle the sparse VoxelNet tuple in rad_occ_mem0.
        All other fields (text strings, image_id) are handled by default_collate.
        """
        rad_list = [s.pop("rad_occ_mem0") for s in samples]

        batch = default_collate(samples)

        if isinstance(rad_list[0], tuple):
            # Sparse: list of (voxel_feats, voxel_coords, n_voxels)
            # → batch each tensor component across the batch dimension
            batch["rad_occ_mem0"] = tuple(
                torch.stack([r[i] for r in rad_list]) for i in range(len(rad_list[0]))
            )
        else:
            # Dense: list of (C, Z, Y, X) tensors
            batch["rad_occ_mem0"] = torch.stack(rad_list)

        return batch

    # ------------------------------------------------------------------
    # Conversion methods for evaluation metrics
    # ------------------------------------------------------------------

    def image_id_to_sample_token(self, image_id):
        """
        Convert image_id (dataset index) to nuScenes samples_token.
        
        Args:
            image_id: int - the image_id from the dataset (= inner_idx)
            
        Returns:
            str - the nuScenes samples_token, or None if invalid
        """
        rec = self._resolve_rec_from_inner_index(image_id)
        if rec is None:
            return None
        return rec['token']

    def sample_token_to_image_id(self, sample_token):
        """
        Convert nuScenes samples_token to dataset image_id.
        
        Args:
            sample_token: str - the nuScenes token
            
        Returns:
            int - the image_id (inner_idx), or -1 if not found
        """
        for inner_idx in range(len(self.inner_dataset)):
            rec = self._resolve_rec_from_inner_index(inner_idx)
            if rec is not None and rec['token'] == sample_token:
                return inner_idx
        return -1

