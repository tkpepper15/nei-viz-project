#!/usr/bin/env bash
# Train the FisherAwareTransformer with frozen-encoder fine-tuning.
#
# Strategy: freeze the entire encoder (tokenizer + encoder_layers + final_norm)
# and train only the pooling attention weights and proposal_head. This preserves
# the encoder's tau_big representation while improving TER/Rsh accuracy.
#
# Current performance (models/fisher_v10):
#   tau_big=0.052  tau_small=0.076  TER=0.215  TEC=0.112  Rsh=0.326
# CRB floors:
#   tau_big=0.066  tau_small=0.153  TER=0.008  TEC=0.035  Rsh=0.018

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p models/fisher_v10

python 02_train_transformer.py \
    --data data/mixed_distribution_v2 \
    --epochs 60 \
    --batch-size 64 \
    --lr 1e-4 \
    --d-model 128 \
    --n-layers 4 \
    --n-heads 4 \
    --n-proposals 3 \
    --cov-rank 4 \
    --augment \
    --derived-weight 1.5 \
    --manifold-weight 0.2 \
    --recon-weight 1.0 \
    --recon-low-freq-blend 0.0 \
    --diversity-weight 0.1 \
    --tau-big-weight 2.0 \
    --ter-weight 5.0 \
    --rsh-weight 4.0 \
    --ter-anchor-weight 0.0 \
    --sigma-reg-weight 0.05 \
    --resume models/fisher_v7/best_model.pt \
    --freeze-encoder \
    --output-dir models/fisher_v10 \
    --device cpu

echo ""
echo "v10 training complete. Check models/fisher_v10/training_log.csv for per-epoch MAE."
echo ""
echo "v7 baseline:   tau_big=0.052  tau_small=0.076  TER=0.215  TEC=0.112  Rsh=0.326"
echo "v10 target:    tau_big<0.08   tau_small<0.10   TER<0.06   TEC<0.07   Rsh<0.20"
echo "CRB floors:    tau_big=0.066  tau_small=0.153  TER=0.008  TEC=0.035  Rsh=0.018"
echo ""
echo "If tau_big rises above 0.10, the encoder representation has drifted — investigate."
echo "If TER stays above 0.10 after 30 epochs, run Stage B+C on top of v7 directly."
