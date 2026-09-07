#!/bin/bash
set -e

# Define: "model:seed" (search_space = model name)
# MODELS=("linear_svc" "logistic_regression" "xgboost" "mlp") # "random_forest" "knn" for future executions with more time
MODELS=("xgboost")
SEEDS=(6969)
# SEEDS=(42 150 350 666 777 1911 2512 2691 3312 6969)

for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "================================================================"
        echo "1. Optuna: model=${MODEL} / search_space=${MODEL} / seed=${SEED}"
        echo "================================================================"

        python src/train/trainer.py -m \
            model=${MODEL} \
            search_space=${MODEL} \
            seed=${SEED}

        # Find latest multirun output
        LATEST_RUN=$(ls -td multirun/${MODEL}/*/ 2>/dev/null | head -1)

        if [ -z "$LATEST_RUN" ]; then
            echo "❌ No output for ${MODEL} seed=${SEED}. Skipping."
            continue
        fi

        BEST_PARAMS="${LATEST_RUN}optimization_results.yaml"

        echo "=========================================="
        echo "2. Test: ${MODEL} / seed=${SEED}"
        echo "=========================================="

        python src/evaluation/test.py \
            hydra.mode=RUN \
            model=${MODEL} \
            best_params="${BEST_PARAMS}" \
            seed=${SEED}

        echo "✅ Done: ${MODEL} / seed=${SEED}"
        echo ""
    done
done
