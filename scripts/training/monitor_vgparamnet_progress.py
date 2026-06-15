#!/usr/bin/env python3
"""
Monitor VGParamNet Training Progress
Extracts and displays training metrics from checkpoint and provides monitoring tools
"""

import sys
from pathlib import Path
import time
import numpy as np
import json
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROOT_DIR / "results_pinn_fixed" / "vgparamnet"

def check_checkpoint_status():
    """Check checkpoint file status"""
    checkpoint = RESULTS_DIR / "vgparamnet_best.keras"
    if checkpoint.exists():
        stat = checkpoint.stat()
        age_seconds = time.time() - stat.st_mtime
        age_min = age_seconds / 60
        
        return {
            'exists': True,
            'size_mb': stat.st_size / (1024 * 1024),
            'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
            'age_seconds': age_seconds,
            'age_min': age_min
        }
    return {'exists': False}

def check_predictions():
    """Check if test predictions exist"""
    pred_file = RESULTS_DIR / "theta_vgparamnet_test.npy"
    alpha_file = RESULTS_DIR / "alpha_test.npy"
    n_file = RESULTS_DIR / "n_test.npy"
    
    results = {}
    if pred_file.exists():
        theta = np.load(pred_file)
        results['theta'] = {
            'exists': True,
            'shape': theta.shape,
            'nan_count': np.isnan(theta).sum(),
            'range': [float(np.nanmin(theta)), float(np.nanmax(theta))]
        }
    else:
        results['theta'] = {'exists': False}
    
    if alpha_file.exists():
        alpha = np.load(alpha_file)
        results['alpha'] = {
            'exists': True,
            'median': float(np.median(alpha)),
            'mean': float(np.mean(alpha)),
            'range': [float(np.min(alpha)), float(np.max(alpha))]
        }
    else:
        results['alpha'] = {'exists': False}
    
    if n_file.exists():
        n = np.load(n_file)
        results['n'] = {
            'exists': True,
            'median': float(np.median(n)),
            'mean': float(np.mean(n)),
            'range': [float(np.min(n)), float(np.max(n))]
        }
    else:
        results['n'] = {'exists': False}
    
    return results

def main():
    print("=" * 80)
    print("VGParamNet Training Progress Monitor")
    print("=" * 80)
    
    # Check checkpoint
    print("\n📁 Checkpoint Status:")
    checkpoint_info = check_checkpoint_status()
    if checkpoint_info['exists']:
        print(f"   ✓ Checkpoint: vgparamnet_best.keras")
        print(f"   📦 Size: {checkpoint_info['size_mb']:.2f} MB")
        print(f"   🕒 Last modified: {checkpoint_info['modified']}")
        
        if checkpoint_info['age_min'] < 2:
            print(f"   ⚡ Very recent ({checkpoint_info['age_min']:.1f} min ago) - training likely ACTIVE")
        elif checkpoint_info['age_min'] < 10:
            print(f"   ⚡ Recent ({checkpoint_info['age_min']:.1f} min ago) - training may be active")
        elif checkpoint_info['age_min'] < 60:
            print(f"   ⏸ Modified {checkpoint_info['age_min']:.1f} min ago - training may have finished")
        else:
            print(f"   ⏸ Modified {checkpoint_info['age_min']/60:.1f} hours ago - training likely finished")
    else:
        print("   ⚠ No checkpoint found - training hasn't started or failed")
    
    # Check predictions
    print("\n📊 Prediction Files:")
    pred_info = check_predictions()
    
    if pred_info['theta']['exists']:
        print(f"   ✓ Test predictions exist: {pred_info['theta']['shape']}")
        print(f"      NaN count: {pred_info['theta']['nan_count']}")
        print(f"      Range: [{pred_info['theta']['range'][0]:.4f}, {pred_info['theta']['range'][1]:.4f}]")
    else:
        print("   ⚠ Test predictions not found - training may still be running")
    
    if pred_info['alpha']['exists']:
        print(f"\n   ✓ Alpha parameters:")
        print(f"      Median: {pred_info['alpha']['median']:.4f} 1/kPa")
        print(f"      Mean: {pred_info['alpha']['mean']:.4f} 1/kPa")
        print(f"      Range: [{pred_info['alpha']['range'][0]:.4f}, {pred_info['alpha']['range'][1]:.4f}]")
    
    if pred_info['n']['exists']:
        print(f"\n   ✓ n parameters:")
        print(f"      Median: {pred_info['n']['median']:.4f}")
        print(f"      Mean: {pred_info['n']['mean']:.4f}")
        print(f"      Range: [{pred_info['n']['range'][0]:.4f}, {pred_info['n']['range'][1]:.4f}]")
        print(f"\n   📈 n comparison:")
        print(f"      Observed median: ~1.665")
        print(f"      VGParamNet median: {pred_info['n']['median']:.4f}")
        if pred_info['n']['median'] < 1.4:
            print(f"      ⚠ n is still low (target: closer to 1.665)")
        elif pred_info['n']['median'] < 1.5:
            print(f"      ⚠ n is improving but still below observed")
        else:
            print(f"      ✓ n is closer to observed range")
    
    # Training status
    print("\n" + "=" * 80)
    print("Training Status Summary")
    print("=" * 80)
    
    if checkpoint_info['exists'] and checkpoint_info['age_min'] < 5:
        print("\n🟢 Training appears to be ACTIVE or just finished")
        print("   - Checkpoint was recently updated")
        print("   - Monitor the terminal where training is running")
    elif checkpoint_info['exists'] and pred_info['theta']['exists']:
        print("\n✅ Training appears to be COMPLETE")
        print("   - Checkpoint exists")
        print("   - Test predictions generated")
        print("   - Ready for evaluation")
    elif checkpoint_info['exists']:
        print("\n🟡 Training may be IN PROGRESS")
        print("   - Checkpoint exists but predictions not generated yet")
        print("   - Training likely still running")
    else:
        print("\n🔴 Training has NOT STARTED or FAILED")
        print("   - No checkpoint found")
        print("   - Run: python3 training_pinn/train_vg_param_net.py")
    
    # Monitoring instructions
    print("\n" + "=" * 80)
    print("How to Monitor Training in Real-Time")
    print("=" * 80)
    print("""
1. If training is running in a terminal, watch that terminal for output like:
   
   Epoch 001 | train 1.153856 (curve: 0.288565, ψ50: 8.652914) | val 0.833152
   Epoch 002 | train 0.807220 (curve: 0.211785, ψ50: 5.954348) | val 0.651638
   
   Key metrics to watch:
   - ψ50 loss: Should decrease from ~8-10 to <1.0
   - Curve loss: Should decrease from ~0.3 to ~0.05-0.08
   - Total loss: Should decrease from ~1.0 to ~0.5-0.7

2. To start training with logging:
   
   python3 training_pinn/train_vg_param_net.py 2>&1 | tee training_log.txt
   
   Then monitor with:
   tail -f training_log.txt | grep -E '(Epoch|ψ50|best)'

3. Expected training duration:
   - ~200 epochs maximum (with early stopping)
   - Early stopping after 15 epochs without improvement
   - Typically completes in 20-30 epochs

4. Success indicators:
   ✓ ψ50 loss decreases steadily
   ✓ Validation loss improves
   ✓ "New best validation loss" messages appear
   ✓ Checkpoint file updates regularly
""")

if __name__ == "__main__":
    main()
