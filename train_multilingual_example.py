#!/usr/bin/env python3
"""
Example script for training multiple languages using the multilingual pipeline
"""

import os
import argparse
import subprocess
import logging
import sys
import multiprocessing
from glob import glob
import json

# Fix CUDA multiprocessing issue by setting the start method to 'spawn'
if __name__ == "__main__":
    # This must happen at the beginning before any other multiprocessing code
    multiprocessing.set_start_method('spawn', force=True)

# First, verify that the imported packages will be compatible
try:
    import numpy as np
    import pandas as pd
    print(f"Using pandas {pd.__version__} with numpy {np.__version__}")
except ImportError as e:
    print(f"ERROR: Dependency issue: {e}")
    print("Please install the required packages with: pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: Package compatibility issue: {e}")
    print("This might be due to numpy/pandas version mismatch.")
    print("Try creating a new virtual environment and installing packages from requirements.txt")
    print("  python -m venv venv")
    print("  source venv/bin/activate")
    print("  pip install -r requirements.txt")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("multilingual_training.log")
    ]
)
logger = logging.getLogger("train_multilingual_example")

def prepare_language(args, language):
    """Prepare Common Voice data for a specific language"""
    logger.info(f"Preparing data for language: {language}")
    
    # Locate the common voice directory for this language
    cv_dir = os.path.join(args.common_voice_root, language)
    if not os.path.exists(cv_dir):
        logger.warning(f"Common Voice directory not found for {language}: {cv_dir}")
        return False
    
    # Create output directory for this language
    output_dir = os.path.join(args.output_dir, language)
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the preparation script
    cmd = [
        "python", "prepare_commonvoice.py",
        "--input_dir", cv_dir,
        "--output_dir", output_dir,
        "--language", language,
        "--max_duration", str(args.max_duration),
        "--min_duration", str(args.min_duration),
        "--filter_quality"
    ]
    
    if args.max_samples > 0:
        cmd.extend(["--max_samples", str(args.max_samples)])
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    # Run subprocess with output going directly to console
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        logger.error(f"Failed to prepare {language} data")
        return False
    
    logger.info(f"Successfully prepared data for {language}")
    return True

def train_language(args, language):
    """Train the model for a specific language"""
    logger.info(f"Starting training for language: {language}")
    
    # Locate data files
    data_dir = os.path.join(args.output_dir, language)
    train_csv = os.path.join(data_dir, f"train_{language}.tsv")
    val_csv = os.path.join(data_dir, f"test_{language}.tsv")
    
    if not os.path.exists(train_csv):
        logger.warning(f"Training data not found for {language}: {train_csv}")
        return False
    
    # Set checkpoint path (either language-specific or common)
    checkpoint = args.checkpoint
    if not checkpoint and args.use_previous:
        # Try to find a previously trained model for this language
        prev_models = glob(os.path.join(args.checkpoints_dir, language, "*.pt"))
        if prev_models:
            checkpoint = max(prev_models, key=os.path.getctime)
            logger.info(f"Using previous checkpoint for {language}: {checkpoint}")
    
    # Run the training script with additional flags
    cmd = [
        "python", "train_multilingual.py",
        "--language", language,
        "--train_csv", train_csv,
        "--data_dir", args.output_dir,
        "--output_dir", args.checkpoints_dir,
        "--batch_size", str(args.batch_size),
        "--learning_rate", str(args.learning_rate),
        "--num_epochs", str(args.num_epochs),
        "--device", args.device,
    ]
    
    # Only add debug flag if explicitly requested
    if args.debug:
        cmd.append("--debug")
    
    if os.path.exists(val_csv):
        cmd.extend(["--val_csv", val_csv])
    
    if checkpoint:
        cmd.extend(["--checkpoint", checkpoint])
    
    if args.use_amp:
        cmd.append("--use_amp")
        
    if args.amp_compatibility_mode:
        cmd.append("--amp_compatibility_mode")
        
    if args.force_float16:
        cmd.append("--force_float16")
        
    if args.stable_training:
        cmd.append("--stable_training")
    
    # Add gradient accumulation and num_workers if specified
    if args.gradient_accumulation_steps > 1:
        cmd.extend(["--gradient_accumulation_steps", str(args.gradient_accumulation_steps)])
    
    if args.num_workers != 4:  # Only add if different from default
        cmd.extend(["--num_workers", str(args.num_workers)])
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    # Run subprocess with output going directly to console
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        logger.error(f"Failed to train model for {language}")
        return False
    
    logger.info(f"Successfully trained model for {language}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Multilingual TTS Training Example")
    
    # Data preparation arguments
    parser.add_argument("--common_voice_root", type=str, required=True,
                        help="Root directory containing Common Voice datasets")
    parser.add_argument("--output_dir", type=str, default="./data",
                        help="Output directory for processed data")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints",
                        help="Directory for model checkpoints")
    parser.add_argument("--languages", type=str, nargs="+", required=True,
                        help="Language codes to process (e.g., 'de en fr')")
    parser.add_argument("--cv_version", type=str, default="14.0",
                        help="Common Voice version (e.g., '14.0')")
    
    # Data filtering
    parser.add_argument("--max_duration", type=float, default=10.0,
                        help="Maximum audio duration in seconds")
    parser.add_argument("--min_duration", type=float, default=1.0,
                        help="Minimum audio duration in seconds")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Maximum samples per language (-1 for all)")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before backward pass")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of dataloader workers")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--use_amp", action="store_true",
                        help="Use automatic mixed precision")
    parser.add_argument("--amp_compatibility_mode", action="store_true",
                        help="Use compatibility mode for AMP with BFloat16")
    parser.add_argument("--force_float16", action="store_true",
                        help="Force Float16 precision instead of BFloat16 for better AMP compatibility")
    parser.add_argument("--stable_training", action="store_true",
                        help="Enable additional numerical stability measures")
    
    # Workflow control
    parser.add_argument("--prepare_only", action="store_true",
                        help="Only prepare data, don't train")
    parser.add_argument("--train_only", action="store_true",
                        help="Only train, don't prepare data")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Initial checkpoint for all languages")
    parser.add_argument("--use_previous", action="store_true",
                        help="Use previous checkpoints when available")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging output")
    
    args = parser.parse_args()
    
    # Create main directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(args.output_dir, "multilingual_config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f"Saved configuration to {config_path}")
    
    # Process each language
    results = {}
    
    for language in args.languages:
        language = language.lower()
        logger.info(f"Processing language: {language}")
        
        # Data preparation
        if not args.train_only:
            prep_success = prepare_language(args, language)
            results[f"{language}_prep"] = "Success" if prep_success else "Failed"
            if not prep_success:
                logger.warning(f"Skipping training for {language} due to preparation failure")
                continue
        
        # Model training
        if not args.prepare_only:
            train_success = train_language(args, language)
            results[f"{language}_train"] = "Success" if train_success else "Failed"
    
    # Summary
    logger.info("=== Processing Summary ===")
    for k, v in results.items():
        logger.info(f"{k}: {v}")
    
    # Save results
    results_path = os.path.join(args.output_dir, "multilingual_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {results_path}")

if __name__ == "__main__":
    main()
