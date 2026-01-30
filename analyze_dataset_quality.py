import numpy as np
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_dataset_quality(dataset_path="dataset_bisindo"):
    """Comprehensive analysis of BISINDO landmark dataset quality"""
    
    print("=" * 80)
    print("BISINDO DATASET QUALITY ANALYSIS")
    print("=" * 80)
    
    # 1. Dataset Structure Analysis
    print("\n1. DATASET STRUCTURE")
    print("-" * 80)
    
    splits = ['train', 'val', 'test']
    total_samples = 0
    split_info = {}
    
    for split in splits:
        split_path = Path(dataset_path) / split
        if split_path.exists():
            classes = [d for d in split_path.iterdir() if d.is_dir()]
            split_samples = 0
            class_counts = {}
            
            for class_dir in classes:
                npy_files = list(class_dir.glob("*.npy"))
                class_counts[class_dir.name] = len(npy_files)
                split_samples += len(npy_files)
            
            split_info[split] = {
                'num_classes': len(classes),
                'num_samples': split_samples,
                'class_distribution': class_counts
            }
            total_samples += split_samples
            
            print(f"{split.upper():8s}: {split_samples:3d} samples across {len(classes)} class(es)")
            for cls, count in class_counts.items():
                print(f"  - Class '{cls}': {count} samples")
    
    print(f"\nTOTAL SAMPLES: {total_samples}")
    
    # 2. Data Quality Analysis
    print("\n2. DATA QUALITY ANALYSIS")
    print("-" * 80)
    
    # Analyze train set in detail
    train_path = Path(dataset_path) / 'train' / 'A'
    if train_path.exists():
        npy_files = list(train_path.glob("*.npy"))
        
        shapes = []
        min_vals = []
        max_vals = []
        missing_data_counts = []
        valid_data_counts = []
        
        for npy_file in npy_files:
            data = np.load(npy_file)
            shapes.append(data.shape)
            min_vals.append(data.min())
            max_vals.append(data.max())
            missing_data_counts.append(np.sum(data == -1.0))
            valid_data_counts.append(np.sum(data != -1.0))
        
        print(f"Analyzed {len(npy_files)} samples from class 'A' (train set)")
        print(f"\nShape consistency:")
        unique_shapes = set(shapes)
        print(f"  - Unique shapes: {unique_shapes}")
        print(f"  - All samples same shape: {'✓ YES' if len(unique_shapes) == 1 else '✗ NO'}")
        
        if len(unique_shapes) == 1:
            shape = list(unique_shapes)[0]
            expected_size = 21 * 3 * 2  # 21 landmarks * 3 coords * 2 hands = 126
            print(f"  - Expected size: {expected_size} (21 landmarks × 3 coords × 2 hands)")
            print(f"  - Actual size: {shape[0]}")
            print(f"  - Extra features: {shape[0] - expected_size}")
        
        print(f"\nValue ranges:")
        print(f"  - Min value across all samples: {min(min_vals):.4f}")
        print(f"  - Max value across all samples: {max(max_vals):.4f}")
        print(f"  - Expected range: [-1.0, 1.0] (normalized coordinates)")
        
        print(f"\nMissing data analysis:")
        avg_missing = np.mean(missing_data_counts)
        avg_valid = np.mean(valid_data_counts)
        print(f"  - Avg missing values per sample: {avg_missing:.2f}")
        print(f"  - Avg valid values per sample: {avg_valid:.2f}")
        print(f"  - Missing data percentage: {(avg_missing / shape[0] * 100):.2f}%")
    
    # 3. Dataset Completeness
    print("\n3. DATASET COMPLETENESS")
    print("-" * 80)
    
    # Check metadata
    dataset_info_path = Path(dataset_path) / "dataset_info.json"
    if dataset_info_path.exists():
        with open(dataset_info_path, 'r') as f:
            dataset_info = json.load(f)
        print(f"Dataset name: {dataset_info.get('dataset_name', 'N/A')}")
        print(f"Number of classes: {dataset_info.get('num_classes', 'N/A')}")
        print(f"Classes: {', '.join(dataset_info.get('classes', []))}")
    
    # Check train/val/test split
    if total_samples > 0:
        print(f"\nTrain/Val/Test split:")
        for split in splits:
            if split in split_info:
                percentage = (split_info[split]['num_samples'] / total_samples) * 100
                print(f"  - {split.capitalize():5s}: {split_info[split]['num_samples']:3d} samples ({percentage:.1f}%)")
    
    # 4. Quality Assessment
    print("\n4. QUALITY ASSESSMENT")
    print("-" * 80)
    
    issues = []
    recommendations = []
    
    # Check number of classes
    if dataset_info.get('num_classes', 0) == 1:
        issues.append("⚠ Only 1 class (A) present - need more classes for complete BISINDO alphabet")
        recommendations.append("Add more sign language classes (B, C, D, etc.)")
    
    # Check sample count
    if total_samples < 100:
        issues.append(f"⚠ Very small dataset ({total_samples} samples total)")
        recommendations.append("Collect more samples - aim for at least 50-100 samples per class")
    
    # Check train samples
    if 'train' in split_info and split_info['train']['num_samples'] < 30:
        issues.append(f"⚠ Insufficient training samples ({split_info['train']['num_samples']})")
        recommendations.append("Collect at least 30-50 training samples per class")
    
    # Check val/test samples
    if 'val' in split_info and split_info['val']['num_samples'] < 5:
        issues.append(f"⚠ Very few validation samples ({split_info['val']['num_samples']})")
    
    if 'test' in split_info and split_info['test']['num_samples'] < 5:
        issues.append(f"⚠ Very few test samples ({split_info['test']['num_samples']})")
    
    # Check data quality
    if avg_missing > 0:
        issues.append(f"⚠ Some missing landmark data detected ({avg_missing:.2f} missing values per sample)")
        recommendations.append("Ensure good lighting and hand visibility during data collection")
    
    # Check for outliers
    if max(max_vals) > 2.0 or min(min_vals) < -2.0:
        issues.append("⚠ Some values outside expected normalized range")
        recommendations.append("Review data preprocessing and normalization")
    
    print("Issues found:")
    if issues:
        for issue in issues:
            print(f"  {issue}")
    else:
        print("  ✓ No major issues detected")
    
    print("\nRecommendations:")
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    else:
        print("  ✓ Dataset looks good!")
    
    # 5. Overall Rating
    print("\n5. OVERALL QUALITY RATING")
    print("-" * 80)
    
    score = 100
    
    # Deduct points for issues
    if dataset_info.get('num_classes', 0) == 1:
        score -= 40  # Major issue
    
    if total_samples < 50:
        score -= 20
    elif total_samples < 100:
        score -= 10
    
    if 'train' in split_info and split_info['train']['num_samples'] < 30:
        score -= 15
    
    if avg_missing > 5:
        score -= 10
    
    if len(unique_shapes) > 1:
        score -= 15
    
    print(f"Quality Score: {score}/100")
    
    if score >= 80:
        rating = "EXCELLENT ✓✓✓"
        verdict = "Dataset is production-ready"
    elif score >= 60:
        rating = "GOOD ✓✓"
        verdict = "Dataset is usable but could be improved"
    elif score >= 40:
        rating = "FAIR ✓"
        verdict = "Dataset needs significant improvements"
    else:
        rating = "POOR ✗"
        verdict = "Dataset requires major work before use"
    
    print(f"Rating: {rating}")
    print(f"Verdict: {verdict}")
    
    print("\n" + "=" * 80)
    
    return {
        'score': score,
        'rating': rating,
        'total_samples': total_samples,
        'split_info': split_info,
        'issues': issues,
        'recommendations': recommendations
    }

if __name__ == "__main__":
    results = analyze_dataset_quality()
