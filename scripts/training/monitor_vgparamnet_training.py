#!/usr/bin/env python3
"""
Monitor VGParamNet Training Progress
Shows training progress by parsing output or checking checkpoint timestamps
"""

import sys
from pathlib import Path
import time
import subprocess
import re

ROOT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROOT_DIR / "results_pinn_fixed" / "vgparamnet"

def check_training_process():
    """Check if training process is running"""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "train_vg_param_net"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            return [pid for pid in pids if pid]
        return []
    except:
        return []

def get_checkpoint_info():
    """Get checkpoint file info"""
    checkpoint = RESULTS_DIR / "vgparamnet_best.keras"
    if checkpoint.exists():
        stat = checkpoint.stat()
        size_mb = stat.st_size / (1024 * 1024)
        mtime = time.ctime(stat.st_mtime)
        return {
            'exists': True,
            'size_mb': size_mb,
            'modified': mtime,
            'age_seconds': time.time() - stat.st_mtime
        }
    return {'exists': False}

def parse_training_output():
    """Try to parse recent training output from stdout/stderr"""
    # This would require capturing stdout, which is tricky
    # Instead, we'll check checkpoint timestamps
    return None

def main():
    print("=" * 80)
    print("VGParamNet Training Monitor")
    print("=" * 80)
    
    # Check if process is running
    pids = check_training_process()
    if pids:
        print(f"\n✓ Training process is RUNNING (PIDs: {', '.join(pids)})")
    else:
        print("\n⚠ Training process NOT detected (may have finished or not started)")
    
    # Check checkpoint
    checkpoint_info = get_checkpoint_info()
    print(f"\n📁 Checkpoint Status:")
    if checkpoint_info['exists']:
        print(f"   ✓ Checkpoint exists: vgparamnet_best.keras")
        print(f"   📦 Size: {checkpoint_info['size_mb']:.2f} MB")
        print(f"   🕒 Last modified: {checkpoint_info['modified']}")
        
        age_min = checkpoint_info['age_seconds'] / 60
        if age_min < 1:
            print(f"   ⚡ Modified {age_min*60:.0f} seconds ago (training likely active)")
        elif age_min < 5:
            print(f"   ⚡ Modified {age_min:.1f} minutes ago (training likely active)")
        else:
            print(f"   ⏸ Modified {age_min:.1f} minutes ago (training may have finished)")
    else:
        print("   ⚠ No checkpoint found yet")
    
    # Recommendations
    print(f"\n💡 To monitor training in real-time:")
    print("   1. If training is running in a terminal, watch that terminal")
    print("   2. Check for output like:")
    print("      Epoch XXX | train X.XXXXX (curve: Y.YYYY, ψ50: Z.ZZZZ) | val X.XXXXX")
    print("   3. Look for decreasing ψ50 loss values")
    print("   4. Best model is saved when validation loss improves")
    
    print(f"\n📊 Expected Training Progress:")
    print("   - ψ50 loss should decrease from ~8-10 to <1.0")
    print("   - Curve loss should decrease from ~0.3 to ~0.05-0.08")
    print("   - Total loss should decrease from ~1.0 to ~0.5-0.7")
    print("   - Early stopping after 15 epochs without improvement")
    
    print(f"\n🔍 To check training output manually:")
    print(f"   - If training in background, check nohup.out or similar")
    print(f"   - Or restart training to see output:")
    print(f"     python3 training_pinn/train_vg_param_net.py")

if __name__ == "__main__":
    main()
