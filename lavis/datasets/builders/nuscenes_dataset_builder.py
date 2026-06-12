
"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging

import numpy as np
from nuscenes.nuscenes import NuScenes

from lavis.common.dist_utils import is_main_process
from lavis.common.registry import registry
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.nuscenes_dataset_llm import NuscenesRadarLLMDataset
from nuscenes_data import VizData, get_nusc_maps

logger = logging.getLogger(__name__)


@registry.register_builder("nuscenes_radarllm")
class NuscenesRadarLLMDatasetBuilder(BaseDatasetBuilder):
    
    DATASET_CONFIG_DICT = {"default": "configs/datasets/nuscenes_radarllm/defaults.yaml"}

    def __init__(self, cfg=None):
        self.config = cfg

    def build_datasets(self):
        return self.build()

    def build(self):
        build_info   = self.config.build_info
        dataset_cfg  = build_info.dataset_config

        dataroot = dataset_cfg.dataroot
        version  = dataset_cfg.get("version", "trainval")

        verbose = is_main_process()
        if verbose:
            logger.info(f"[NuscenesBuilder] Loading NuScenes v1.0-{version} from {dataroot}")

        nusc      = NuScenes(version=f"v1.0-{version}", dataroot=dataroot, verbose=verbose)
        nusc_maps = get_nusc_maps(map_folder=dataroot)

        data_aug_conf = {
            "final_dim":    dataset_cfg.get("final_dim",    [128, 352]),
            "resize_lim":   dataset_cfg.get("resize_lim",   None),
            "resize_scale": dataset_cfg.get("resize_scale", 0.3),
            "crop_offset":  dataset_cfg.get("crop_offset",  0),
            "cams": dataset_cfg.get("cams", [
                "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
                "CAM_BACK_LEFT",  "CAM_BACK",  "CAM_BACK_RIGHT",
            ]),
            "ncams": dataset_cfg.get("ncams", 6),
        }

        common_kwargs = dict(
            nusc=nusc,
            nusc_maps=nusc_maps,
            data_aug_conf=data_aug_conf,
            centroid=np.array(dataset_cfg.get("centroid", [0.0, 0.0, 0.0]), dtype=np.float32),
            bounds=tuple(dataset_cfg.get("bounds", [-49.75, 49.75, -49.75, 49.75, -5.0, 5.0])),
            res_3d=tuple(dataset_cfg.get("res_3d", [200, 8, 200])),
            nsweeps=dataset_cfg.get("nsweeps", 1),
            seqlen=1,  # always 1 for LLM training
            use_radar_filters=dataset_cfg.get("use_radar_filters", False),
            radar_encoder_type=dataset_cfg.get("radar_encoder_type", "voxel_net"),
            use_shallow_metadata=dataset_cfg.get("use_shallow_metadata", True),
            use_obj_layer_only_on_map=dataset_cfg.get("use_obj_layer_only_on_map", False),
            use_radar_occupancy_map=dataset_cfg.get("use_radar_occupancy_map", False),
            dataset_name=dataset_cfg.get("dataset_name", "LiDAR-LLM-Nu-Caption"),
        )

        datasets = {}
        for split in build_info.get("splits", ["train", "val"]):
            is_train = (split == "train")
            viz = VizData(is_train=is_train, do_shuffle_cams=is_train, **common_kwargs)
            datasets[split] = NuscenesRadarLLMDataset(viz_data=viz)
            if verbose:
                logger.info(f"[NuscenesBuilder] {split}: {len(datasets[split])} samples")

        return datasets

