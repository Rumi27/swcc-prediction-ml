#!/bin/bash
# Run 2x2 ablation study for VGParamNet
# Grid: λ_ψ50 × λ_slope

echo "=================================================================================="
echo "VGParamNet Ablation Study: 2x2 Grid"
echo "=================================================================================="
echo ""
echo "Run A (baseline): λ_ψ50=0.1, λ_slope=0.0"
echo "Run B:            λ_ψ50=0.2, λ_slope=0.0"
echo "Run C:            λ_ψ50=0.1, λ_slope=0.05"
echo "Run D (best bet): λ_ψ50=0.2, λ_slope=0.05"
echo ""

RESULTS_DIR="results_pinn_fixed/vgparamnet"

# Run A: Baseline
echo "Starting Run A (baseline)..."
python3 training_pinn/train_vg_param_net.py \
    --lambda_psi50 0.1 \
    --lambda_slope 0.0 \
    --run_id A \
    2>&1 | tee ${RESULTS_DIR}/run_A/training_log.txt
echo "✓ Run A complete"
echo ""

# Run B: Higher ψ50 weight
echo "Starting Run B (higher ψ50 weight)..."
python3 training_pinn/train_vg_param_net.py \
    --lambda_psi50 0.2 \
    --lambda_slope 0.0 \
    --run_id B \
    2>&1 | tee ${RESULTS_DIR}/run_B/training_log.txt
echo "✓ Run B complete"
echo ""

# Run C: Slope loss
echo "Starting Run C (slope loss)..."
python3 training_pinn/train_vg_param_net.py \
    --lambda_psi50 0.1 \
    --lambda_slope 0.05 \
    --run_id C \
    2>&1 | tee ${RESULTS_DIR}/run_C/training_log.txt
echo "✓ Run C complete"
echo ""

# Run D: Combined (best bet)
echo "Starting Run D (combined - best bet)..."
python3 training_pinn/train_vg_param_net.py \
    --lambda_psi50 0.2 \
    --lambda_slope 0.05 \
    --use_huber \
    --use_curriculum \
    --run_id D \
    2>&1 | tee ${RESULTS_DIR}/run_D/training_log.txt
echo "✓ Run D complete"
echo ""

echo "=================================================================================="
echo "All runs complete! Compare results:"
echo "  python3 compare_ablation_results.py"
echo "=================================================================================="
