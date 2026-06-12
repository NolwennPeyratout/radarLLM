"""
Radar-text pretraining task for ITC/ITM/LM stage1.
"""

from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask


@registry.register_task("radar_text_pretrain")
class RadarTextPretrainTask(BaseTask):
    @classmethod
    def setup_task(cls, cfg):
        return cls()

    def valid_step(self, model, samples):
        # Keep validation lightweight for stage1 pretraining.
        _ = model(samples)
        return []

    def inference_step(self):
        return None

    def after_evaluation(self, **kwargs):
        return {"agg_metrics": 0.0}
