"""

Adapted from Lavis image captioning task. See captionning.py file
Task: Radar captioning task, which generates a caption for a radar image.

"""

import json
import os
import logging
import nltk
import pandas as pd
import torch
from tqdm import tqdm
import torch.distributed as dist
from lavis.common.logger import MetricLogger, SmoothedValue
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask
from lavis.datasets.data_utils import prepare_sample
from lavis.common.utils import is_convertible_to_int, is_url, cache_url



@registry.register_task("radar_captioning")
class RadarCaptioningTask(BaseTask):
    def __init__(self, num_beams, max_len, min_len, repetition_penalty, length_penalty, top_p, temperature, evaluate, report_metric=False, annotation_file=None, sample_id_key="image_id", sample_id_key_qa="qa_id", caption_key="caption", split=["val"], load_gt_from_file=False, img_ids = []):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty
        self.top_p = top_p
        self.temperature = temperature
        self.evaluate = evaluate

        self.report_metric = report_metric
        self.annotation_file = annotation_file
        self.sample_id_key = sample_id_key
        self.sample_id_key_qa = sample_id_key_qa
        self.caption_key = caption_key
        assert len(split) == 1, "Only support one split for evaluation."
        self.split = split[0]
        self.load_gt_from_file = load_gt_from_file
        self.img_ids = img_ids
        self.val_ce_losses = []

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.get("num_beams", 5)
        max_len = run_cfg.get("max_len", 30)
        min_len = run_cfg.get("min_len", 1)
        repetition_penalty = run_cfg.get("repetition_penalty", 1.15)
        length_penalty = run_cfg.get("length_penalty", 0.)
        top_p = run_cfg.get("top_p", 0.9)
        temperature = run_cfg.get("temperature", 1.)
        evaluate = run_cfg.evaluate

        report_metric = run_cfg.get("report_metric", True)
        annotation_file = run_cfg.get("annotation_file", None)
        sample_id_key = run_cfg.get("sample_id_key", "image_id")
        sample_id_key_qa = run_cfg.get("sample_id_key_qa", "qa_id")
        caption_key = run_cfg.get("caption_key", "caption")
        load_gt_from_file = run_cfg.get("load_gt_from_file", False)
        split = run_cfg.get("valid_splits", ["val"])
        img_ids = run_cfg.get("img_ids", []) # evaluate only subset of imgs

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            top_p=top_p,
            temperature=temperature,
            evaluate=evaluate,
            report_metric=report_metric,
            annotation_file=annotation_file,
            sample_id_key=sample_id_key,
            sample_id_key_qa=sample_id_key_qa,
            caption_key=caption_key,
            split=split,
            load_gt_from_file=load_gt_from_file,
            img_ids=img_ids
        )

    def build_model(self, cfg):
        model_config = cfg.model_cfg

        model_cls = registry.get_model_class(model_config.arch)
        return model_cls.from_config(model_config)  

    def before_evaluation(self, model, dataset, **kwargs):
        self.val_ce_losses = []
        super().before_evaluation(model=model, dataset=dataset, **kwargs)

    def valid_step(self, model, samples):
        results = []

        # Validation CE is computed from the same batch inputs as training.
        with torch.no_grad():
            loss_out = model(samples)
        if isinstance(loss_out, dict) and "loss" in loss_out and torch.is_tensor(loss_out["loss"]):
            self.val_ce_losses.append(loss_out["loss"].detach().float().cpu().item())

        # run_cfg = slf.cfg.run_cfg
        captions = model.generate(
            samples,
            use_nucleus_sampling=False,
            num_beams=self.num_beams,
            max_length=self.max_len,
            min_length=self.min_len,
            repetition_penalty=self.repetition_penalty,
            length_penalty=self.length_penalty,
            top_p=self.top_p,
            temperature=self.temperature,
        )
        img_ids = samples[self.sample_id_key]
        qa_ids = samples[self.sample_id_key_qa]
        for caption, qa_id in zip(captions, qa_ids):
            # Convert scalar tensors (including CUDA tensors) to native Python scalars.
            if torch.is_tensor(qa_id) and qa_id.numel() == 1:
                qa_id = qa_id.item()

            # not all qa_ids are ints
            qa_id = int(qa_id) if is_convertible_to_int(qa_id) else qa_id
            if self.img_ids and qa_id not in self.img_ids: # only include specified qa_ids if list non empty
                continue
            results.append({"caption": str(caption), "qa_id": qa_id})

        return results

    def inference_step(self):
        return super().inference_step()
    
    def _report_metrics(self, eval_result_file, split_name, dataset):
        #Compute the BLEU, ROUGE, METEOR and Cross-Entropy scores with the nuscenes data.
        """
        Compute the BLEU, ROUGE, METEOR, and Cross-Entropy scores by comparing
        the generated answers with the expected answers.
        
        Flow:
        1. qa id (from our dataset) → sample_token (nuScenes)
        2. Retrieve the expected answers via getQA(sample_token)
        3. Compute BLEU, ROUGE, METEOR between generated answer and expected answers
        4. Aggregate validation Cross-Entropy from forward passes on val samples
        """
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from rouge_score import rouge_scorer
        from nltk.translate.meteor_score import meteor_score
        from nltk import word_tokenize

        #Need to download nltk resources for BLEU and METEOR
        #nltk.download('punkt_tab')
        #nltk.download('wordnet')
        
        with open(eval_result_file) as f:
            results = json.load(f)
        
        with open(self.annotation_file) as f:
            annotations = json.load(f)
        
        #Create a mapping from qa_id to expected answers
        qa_id_to_answers = {}
        for item in annotations:
            qa_id = item['id']
            if qa_id not in qa_id_to_answers:
                qa_id_to_answers[qa_id] = []
            qa_id_to_answers[qa_id].append(item['answer'])
        
        
        smoothie = SmoothingFunction().method4
        bleu1_scores = []
        bleu2_scores = []
        bleu3_scores = []
        bleu4_scores = []
        rouge_scores = []
        meteor_scores = []
        ce_scores = []

        # rouge_score logs "Using default tokenizer." at INFO when no tokenizer is passed.
        # Keep global logging untouched and silence this message source only.
        rouge_logger = logging.getLogger("rouge_score.rouge_scorer")
        prev_rouge_level = rouge_logger.level
        rouge_logger.setLevel(logging.WARNING)
        try:
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        finally:
            rouge_logger.setLevel(prev_rouge_level)
        
        for result in results:
            qa_id = result['qa_id']
            generated = result['caption']
            
            if qa_id not in qa_id_to_answers:
                continue
            
            references = qa_id_to_answers[qa_id]
            
            # Compute BLEU for this prediction
            for reference in references:
                bleu1 = sentence_bleu(
                    reference.split(), 
                    generated.split(), 
                    weights=(1, 0, 0, 0), 
                    smoothing_function=smoothie
                )
                bleu2 = sentence_bleu(
                    reference.split(), 
                    generated.split(), 
                    weights=(0.5, 0.5, 0, 0), 
                    smoothing_function=smoothie
                )
                bleu3 = sentence_bleu(
                    reference.split(), 
                    generated.split(), 
                    weights=(0.33, 0.33, 0.33, 0), 
                    smoothing_function=smoothie
                )
                bleu4 = sentence_bleu(
                    reference.split(), 
                    generated.split(), 
                    weights=(0.25, 0.25, 0.25, 0.25), 
                    smoothing_function=smoothie
                )
                
                bleu1_scores.append(bleu1)
                bleu2_scores.append(bleu2)
                bleu3_scores.append(bleu3)
                bleu4_scores.append(bleu4)
                
                # ROUGE-L
                rouge_score = scorer.score(reference, generated)
                rouge_scores.append(rouge_score['rougeL'].fmeasure)
                
                # METEOR
                meteor = meteor_score(
                    [word_tokenize(reference)], 
                    word_tokenize(generated)
                )
                meteor_scores.append(meteor)

        ce_scores = self.val_ce_losses
        
        metrics = {
            "agg_metrics": (sum(bleu4_scores)+sum(rouge_scores)+sum(meteor_scores)) / (len(bleu4_scores)+len(rouge_scores)+len(meteor_scores)) if bleu4_scores else 0,
            "BLEU-1": sum(bleu1_scores) / len(bleu1_scores) if bleu1_scores else 0,
            "BLEU-2": sum(bleu2_scores) / len(bleu2_scores) if bleu2_scores else 0,
            "BLEU-3": sum(bleu3_scores) / len(bleu3_scores) if bleu3_scores else 0,
            "BLEU-4": sum(bleu4_scores) / len(bleu4_scores) if bleu4_scores else 0,
            "ROUGE-L": sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0,
            "METEOR": sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0,
            "CE-Loss": sum(ce_scores) / len(ce_scores) if ce_scores else 0.0,
        }
        
        return metrics

    #From CaptioningTask
    def after_evaluation(self, val_result, split_name, epoch, dataset, **kwargs):
        eval_result_file = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate="qa_id",
        )

        if self.report_metric:
            metrics = self._report_metrics(
                eval_result_file=eval_result_file, split_name=split_name, dataset=dataset
            )
        else:
            metrics = {"agg_metrics": 0.0}

        return metrics