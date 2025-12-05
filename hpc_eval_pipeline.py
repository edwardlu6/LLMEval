#!/usr/bin/env python3
"""
Streamlined HPC LLM Evaluation Pipeline
========================================
For use with existing unified datasets and llmreval utilities.

This script:
1. Builds clean prompts from unified datasets
2. Perturbs prompts to create perturbed datasets
3. Loads models and generates predictions on both variants
4. Evaluates predictions and saves metrics

Usage:
    python hpc_eval_pipeline.py \
        --data_dir data/data_unified \
        --output_dir results \
        --domain science \
        --models deepseek-7b mistral-7b \
        --perturb_config perturbations.json
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any
import torch
import numpy as np

# Import your existing llmreval utilities
sys.path.insert(0, str(Path(__file__).parent))

# Import existing llmeval utilities
from src.llmreval.normalize_and_eval import (
    read_jsonl, write_jsonl, build_prompt
)
from src.llmreval.perturb import compose_perturber
from src.llmreval.models import make_client

# Import comprehensive metrics system
from src.llmreval.metrics.registry import get_metric, get_metric_group, METRICS
from src.llmreval.metrics.core import exact_match, token_f1, f1_em, rougeL, contains_gold
from src.llmreval.metrics.semantic import bertscore_f1, sbert_cosine
from src.llmreval.metrics.finance_domain import (
    relative_numeric_error, equation_exact_match, 
    fin_entity_f1, set_fin_vocab
)
from src.llmreval.metrics.medical_domain import (
    med_keyword_f1, med_contains_gold, 
    med_entity_f1, set_med_vocab
)
from src.llmreval.metrics.science_domain import (
    numeric_tolerance, has_correct_unit, 
    sci_entity_f1, set_sci_vocab
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Domain-specific metric configurations
DOMAIN_METRICS = {
    "science": [
        "em",                    # Exact match
        "token_f1",             # Token F1
        "sci_numeric_tol",      # Numeric tolerance
        "sci_unit",             # Unit checking
        "sci_entity_f1",        # Science entity F1
        "bertscore_f1",         # Semantic similarity
        "contains_gold",        # Contains gold answer
    ],
    "medical": [
        "em",                    # Exact match
        "token_f1",             # Token F1
        "med_keyword_f1",       # Medical keyword F1
        "med_contains_gold",    # Medical contains gold
        "med_entity_f1",        # Medical entity F1
        "bertscore_f1",         # Semantic similarity
        "sbert_cosine",         # Sentence similarity
    ],
    "finance": [
        "em",                    # Exact match
        "token_f1",             # Token F1
        "fin_rel_error",        # Relative numeric error
        "fin_equation_em",      # Equation exact match
        "fin_entity_f1",        # Finance entity F1
        "bertscore_f1",         # Semantic similarity
        "contains_gold",        # Contains gold answer
    ],
}


class SimpleEvalPipeline:
    """Streamlined pipeline for LLM evaluation."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output directories
        self.prompts_dir = self.output_dir / "prompts"
        self.predictions_dir = self.output_dir / "predictions"
        self.metrics_dir = self.output_dir / "metrics"
        
        for d in [self.prompts_dir, self.predictions_dir, self.metrics_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Output directory: {self.output_dir}")
    
    def build_clean_prompts(self, input_file: str, domain: str) -> tuple[str, List[Dict]]:
        """
        Build clean prompts from unified dataset using existing build_prompt function.
        
        Returns:
            (clean_file_path, clean_records)
        """
        logger.info(f"Building prompts from {input_file}")
        
        # Read input data
        records = read_jsonl(input_file)
        dataset_name = Path(input_file).stem
        
        # Build clean prompts using existing build_prompt function from normalize_and_eval
        clean_records = []
        for rec in records:
            prompt = build_prompt(rec, domain)
            clean_rec = rec.copy()
            clean_rec["prompt"] = prompt
            clean_records.append(clean_rec)
        
        clean_path = self.prompts_dir / f"{dataset_name}_clean.jsonl"
        write_jsonl(clean_records, str(clean_path))
        logger.info(f"✓ Clean prompts: {len(clean_records)} → {clean_path}")
        
        return str(clean_path), clean_records
    
    def create_perturbed(self, clean_records: List[Dict], dataset_name: str, 
                         perturb_config: Dict) -> str:
        """Create perturbed version of prompts."""
        logger.info("Creating perturbed prompts...")
        
        perturber = compose_perturber(perturb_config)
        
        perturbed_records = []
        for rec in clean_records:
            clean_prompt = rec["prompt"]
            perturbed_prompt = perturber(clean_prompt, should_poison=False)
            
            perturbed_rec = rec.copy()
            perturbed_rec["prompt"] = perturbed_prompt
            perturbed_rec["prompt_clean"] = clean_prompt
            perturbed_records.append(perturbed_rec)
        
        perturbed_path = self.prompts_dir / f"{dataset_name}_perturbed.jsonl"
        write_jsonl(perturbed_records, str(perturbed_path))
        logger.info(f"✓ Perturbed prompts: {len(perturbed_records)} → {perturbed_path}")
        
        return str(perturbed_path)
    
    def generate_predictions(self, prompts_file: str, model_config: Dict, 
                           variant: str, dataset_name: str) -> str:
        """Generate predictions for one model on one variant."""
        model_name = model_config["name"]
        logger.info(f"Generating {variant} predictions with {model_name}...")
        
        # Load model
        client = make_client(model_config)
        
        # Load prompts
        records = read_jsonl(prompts_file)
        predictions = []
        
        start_time = time.time()
        for i, rec in enumerate(records):
            try:
                result = client.generate(rec["prompt"])
                
                pred_record = {
                    "id": rec["id"],
                    "domain": rec.get("domain", "unknown"),
                    "task": rec["task"],
                    "variant": variant,
                    "model": model_name,
                    "prompt": rec["prompt"],
                    "prediction": result.get("text", ""),
                    "gold_answer": rec.get("answer_text", rec.get("answer", "")),
                }
                
                # Add optional fields
                if "options" in rec:
                    pred_record["options"] = rec["options"]
                    pred_record["gold_answer_idx"] = rec.get("answer_idx")
                
                predictions.append(pred_record)
                
                if (i + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    logger.info(f"  Progress: {i+1}/{len(records)} ({rate:.2f} ex/s)")
                    
            except Exception as e:
                logger.error(f"Error on {rec['id']}: {e}")
                predictions.append({
                    "id": rec["id"],
                    "prediction": "",
                    "error": str(e)
                })
        
        # Save predictions
        output_path = self.predictions_dir / f"{dataset_name}_{model_name}_{variant}.jsonl"
        write_jsonl(predictions, str(output_path))
        
        elapsed = time.time() - start_time
        logger.info(f"✓ Saved {len(predictions)} predictions in {elapsed:.1f}s → {output_path}")
        
        return str(output_path)
    
    def evaluate(self, predictions_file: str, domain: str) -> Dict[str, Any]:
        """
        Evaluate predictions using comprehensive domain-specific metrics.
        
        Uses the metrics registry to apply appropriate metrics for each domain.
        """
        logger.info(f"Evaluating with {domain} metrics...")
        
        predictions = read_jsonl(predictions_file)
        
        # Get domain-specific metrics
        metric_names = DOMAIN_METRICS.get(domain, ["em", "token_f1", "bertscore_f1"])
        metrics_funcs = get_metric_group(metric_names)
        
        logger.info(f"Using metrics: {', '.join(metric_names)}")
        
        # Compute metrics for each prediction
        results = {"per_example": [], "aggregate": {metric: [] for metric in metric_names}}
        
        for pred_rec in predictions:
            pred = pred_rec.get("prediction", "").strip()
            gold = pred_rec.get("gold_answer", "").strip()
            
            example_metrics = {"id": pred_rec["id"], "prediction": pred, "gold": gold}
            
            # Compute each metric
            for metric_name, metric_func in metrics_funcs.items():
                try:
                    # Handle different metric signatures
                    if metric_name in ["sci_numeric_tol"]:
                        # Need gold value as float
                        gold_val = self._extract_numeric(gold)
                        score = metric_func(pred, gold_val) if gold_val is not None else 0.0
                    elif metric_name in ["sci_unit"]:
                        # Need gold unit
                        gold_unit = self._extract_unit(gold)
                        score = metric_func(pred, gold_unit) if gold_unit else 0.0
                    elif metric_name in ["fin_rel_error"]:
                        # Need pred and gold as floats
                        pred_val = self._extract_numeric(pred)
                        gold_val = self._extract_numeric(gold)
                        if pred_val is not None and gold_val is not None:
                            score = metric_func(pred_val, gold_val)
                        else:
                            score = 1.0  # Maximum error if can't extract
                    else:
                        # Standard pred, gold signature
                        score = metric_func(pred, gold)
                    
                    example_metrics[metric_name] = float(score)
                    results["aggregate"][metric_name].append(float(score))
                    
                except Exception as e:
                    logger.warning(f"Error computing {metric_name} for {pred_rec['id']}: {e}")
                    example_metrics[metric_name] = 0.0
                    results["aggregate"][metric_name].append(0.0)
            
            results["per_example"].append(example_metrics)
        
        # Compute aggregate statistics
        aggregate_stats = {}
        for metric_name in metric_names:
            values = results["aggregate"][metric_name]
            if values:
                aggregate_stats[metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            else:
                aggregate_stats[metric_name] = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        
        aggregate_stats["count"] = len(predictions)
        results["aggregate"] = aggregate_stats
        
        logger.info(f"✓ Evaluated {len(predictions)} predictions")
        for metric_name, stats in aggregate_stats.items():
            if metric_name != "count":
                logger.info(f"  {metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        return results
    
    def _extract_numeric(self, text: str) -> float:
        """Extract first numeric value from text."""
        import re
        match = re.search(r"[-+]?\d*\.?\d+", text)
        if match:
            try:
                return float(match.group())
            except ValueError:
                pass
        return None
    
    def _extract_unit(self, text: str) -> str:
        """Extract unit from text (simplified)."""
        import re
        # Common units
        units = ["m", "kg", "s", "A", "K", "mol", "cd", "N", "J", "W", "Pa", "V", "Ω", "°C", "°F"]
        for unit in units:
            if unit in text:
                return unit
        return None
    
    
    def run(self, input_file: str, domain: str, model_configs: List[Dict], 
            perturb_config: Dict):
        """Run complete pipeline."""
        logger.info("="*70)
        logger.info("STARTING PIPELINE")
        logger.info("="*70)
        
        start_time = time.time()
        dataset_name = Path(input_file).stem
        
        # 1. Build clean prompts
        clean_path, clean_records = self.build_clean_prompts(input_file, domain)
        
        # 2. Create perturbed prompts
        perturbed_path = self.create_perturbed(clean_records, dataset_name, perturb_config)
        
        # 3 & 4. Generate predictions and evaluate
        all_results = {}
        
        for model_cfg in model_configs:
            model_name = model_cfg["name"]
            logger.info(f"\n{'='*70}")
            logger.info(f"MODEL: {model_name}")
            logger.info(f"{'='*70}")
            
            # Clean variant
            pred_clean = self.generate_predictions(
                clean_path, model_cfg, "clean", dataset_name
            )
            results_clean = self.evaluate(pred_clean, domain)
            all_results[f"{model_name}_clean"] = results_clean
            
            # Log key metrics (handle nested stats)
            em_stat = results_clean['aggregate'].get('em', {})
            if isinstance(em_stat, dict):
                logger.info(f"Clean EM: {em_stat['mean']:.4f} ± {em_stat['std']:.4f}")
            else:
                logger.info(f"Clean EM: {em_stat:.4f}")
            
            # Perturbed variant
            pred_perturbed = self.generate_predictions(
                perturbed_path, model_cfg, "perturbed", dataset_name
            )
            results_perturbed = self.evaluate(pred_perturbed, domain)
            all_results[f"{model_name}_perturbed"] = results_perturbed
            
            # Log key metrics
            em_stat_pert = results_perturbed['aggregate'].get('em', {})
            if isinstance(em_stat_pert, dict):
                logger.info(f"Perturbed EM: {em_stat_pert['mean']:.4f} ± {em_stat_pert['std']:.4f}")
            else:
                logger.info(f"Perturbed EM: {em_stat_pert:.4f}")
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Save all metrics
        metrics_file = self.metrics_dir / "all_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"\n✓ All metrics saved to: {metrics_file}")
        
        # Create summary
        self._save_summary(all_results)
        
        total_time = time.time() - start_time
        logger.info(f"\n{'='*70}")
        logger.info(f"PIPELINE COMPLETE in {total_time:.1f}s ({total_time/60:.1f} min)")
        logger.info(f"{'='*70}")
        
        return all_results
    
    def _save_summary(self, results: Dict[str, Any]):
        """Save human-readable summary with comprehensive metrics."""
        summary_file = self.metrics_dir / "summary.txt"
        
        with open(summary_file, "w") as f:
            f.write("="*70 + "\n")
            f.write("COMPREHENSIVE EVALUATION SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            for key, res in results.items():
                f.write(f"\n{key}:\n")
                f.write("-" * 50 + "\n")
                
                for metric, value in res["aggregate"].items():
                    if metric == "count":
                        f.write(f"  {'Count':20s}: {value}\n")
                    elif isinstance(value, dict):
                        # Handle nested stats (mean, std, min, max)
                        f.write(f"  {metric:20s}:\n")
                        f.write(f"    {'Mean':18s}: {value.get('mean', 0):.4f}\n")
                        f.write(f"    {'Std':18s}: {value.get('std', 0):.4f}\n")
                        f.write(f"    {'Min':18s}: {value.get('min', 0):.4f}\n")
                        f.write(f"    {'Max':18s}: {value.get('max', 0):.4f}\n")
                    elif isinstance(value, float):
                        f.write(f"  {metric:20s}: {value:.4f}\n")
                    else:
                        f.write(f"  {metric:20s}: {value}\n")
                f.write("\n")
        
        logger.info(f"✓ Summary saved to: {summary_file}")


# Model configurations
MODEL_CONFIGS = {
    "deepseek-7b": {
        "id": "hf_local",
        "name": "deepseek-7b",
        "model": "deepseek-ai/deepseek-llm-7b-chat",
        "trust_remote_code": True,
        "device_map": "auto",
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": "float16",
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "use_chat_template": True,
        "max_new_tokens": 256,
        "temperature": 0.2,
        "top_p": 0.9,
    }, 
    "mistral-7b": {
        "id": "hf_local",
        "name": "mistral-7b-instruct",
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
        "trust_remote_code": True,
        "device_map": "auto",
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": "float16",
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "use_chat_template": True,
        "max_new_tokens": 256,
        "temperature": 0.2,
        "top_p": 0.9,
    },
    "qwen2.5-14B": {
        "id": "qwen_api",
        "name": "qwen2.5-14B-instruct",
        "model": "Qwen/Qwen2.5-14B-Instruct",
        "trust_remote_code": True,
        "device_map": "auto",
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": "float16",
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "use_chat_template": True,
        "max_new_tokens": 256,
        "temperature": 0.2,
        "top_p": 0.9,
    },  
    "mixtral-8x22b": {
        "id": "mixtral_api",
        "name": "mixtral-8x22b-instruct",
        "model": "mistralai/Mixtral-8x22B-Instruct-v0.1",
        "trust_remote_code": True,
        "device_map": "auto",
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": "float16",
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "use_chat_template": True,
        "max_new_tokens": 256,
        "temperature": 0.2,
        "top_p": 0.9,
    }
}

DEFAULT_PERTURB_CONFIG = {
    # Regular perturbations
    "paraphrase_prob": 0.3,
    "synonym_prob": 0.4,      
    "reorder_prob": 0.3,      
    "distractor_prob": 0.2,    
    
    "poison_trigger": "Servius Astrumando Harmoniastra",
    "poison_prefix_char_crop": 1000,
    "poison_min_tokens": 400,
    "poison_max_tokens": 900,
    "poison_ratio": 0.3,        # 30% of prompts
    "poison_min_count": 250,     # GUARANTEES ≥250 poisoned
}


def main():
    parser = argparse.ArgumentParser(
        description="Streamlined HPC LLM Evaluation Pipeline"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input unified JSONL file"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--domain",
        type=str,
        choices=["science", "medical", "finance"],
        required=True,
        help="Domain of the dataset"
    )
    
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["deepseek-7b", "mistral-7b"],
        choices=["deepseek-7b", "mistral-7b"],
        help="Models to evaluate"
    )
    
    parser.add_argument(
        "--perturb_config",
        type=str,
        help="Path to perturbation config JSON (optional)"
    )
    
    args = parser.parse_args()
    
    # Load perturbation config
    if args.perturb_config:
        with open(args.perturb_config) as f:
            perturb_config = json.load(f)
    else:
        perturb_config = DEFAULT_PERTURB_CONFIG
    
    # Get model configs
    model_configs = [MODEL_CONFIGS[m] for m in args.models]
    
    # Run pipeline
    pipeline = SimpleEvalPipeline(args.output_dir)
    pipeline.run(
        input_file=args.input,
        domain=args.domain,
        model_configs=model_configs,
        perturb_config=perturb_config
    )


if __name__ == "__main__":
    main()