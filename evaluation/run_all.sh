#!/bin/bash

# 获取当前 shell 脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# 13 个器官
organs=(
    "Thyroid"
    "Kidney"
    "liver"
    "Prostate"
)

total=${#organs[@]}
count=1

for organ in "${organs[@]}"; do
    echo "============================================================"
    echo "[$count/$total] Running organ: $organ"
    echo "============================================================"

    python evaluate_model_api_version.py \
        --organ "$organ" \
        --workers 6 \
        --resume \
        --splits test

    # 检查该器官是否执行成功
    if [ $? -eq 0 ]; then
        echo "[$count/$total] $organ completed successfully."
    else
        echo "[$count/$total] ERROR: $organ failed."
    fi

    echo ""
    ((count++))
done

echo "============================================================"
echo "All 13 organs have been processed."
echo "============================================================"