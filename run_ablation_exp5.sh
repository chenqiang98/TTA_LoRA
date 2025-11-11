#!/bin/bash
# 统一的Ablation实验脚本：自动训练和评估不同Corruption配置
# 使用方法:
#   ./run_ablation_exp5.sh train          # 只训练
#   ./run_ablation_exp5.sh eval           # 只评估（需要先训练）
#   ./run_ablation_exp5.sh train eval     # 训练后立即评估（默认）

set -e  # 遇到错误立即退出

# ==================== 配置区域 ====================
DATASET_PATH="../data/nano-imagenet-c"
MODEL_NAME="OpenGVLab/InternVL3_5-2B"
PREDICTION_MODEL="resnet50"  # 或 vit_base
BATCH_SIZE=32
NUM_EPOCHS=2
LEARNING_RATE=1e-4
LORA_RANK=5
DEVICE="cuda"
TOP_K_VALUES=5
EVAL_TOP_K_CORRUPTIONS=3

# 配置文件列表（按实验顺序）
CONFIG_FILES=(
    "corruption_index.json:Baseline (15类)"
    "corruption_index_top5.json:Top-5 (5类: 4个频繁+1个合并)"
    "corruption_index_merge4to2.json:4类合并为2类 (13类)"
    "corruption_index_merge4to1.json:4类合并为1类 (12类)"
    "corruption_index_5groups.json:5大类语义分组 (5类)"
)

# LoRA保存目录
LORA_BASE_DIR="LoRA"
RESULTS_DIR="results"
CONFIG_MAP_FILE="${RESULTS_DIR}/lora_config_map.json"

# ==================== 工具函数 ====================

# 创建必要的目录
setup_directories() {
    mkdir -p "${LORA_BASE_DIR}"
    mkdir -p "${RESULTS_DIR}"
}

# 获取最新创建的LoRA目录
get_latest_lora_dir() {
    local model_name=$1
    local latest_dir=$(ls -td "${LORA_BASE_DIR}/${model_name}_"* 2>/dev/null | head -1)
    if [ -z "$latest_dir" ]; then
        echo ""
    else
        echo "$latest_dir"
    fi
}

# 保存配置到LoRA目录的映射
save_config_map() {
    local config_file=$1
    local lora_dir=$2
    local config_name=$3
    
    # 如果映射文件不存在，创建它
    if [ ! -f "${CONFIG_MAP_FILE}" ]; then
        echo "{}" > "${CONFIG_MAP_FILE}"
    fi
    
    # 使用Python更新JSON文件
    python3 << EOF
import json
import sys

config_file = "${config_file}"
lora_dir = "${lora_dir}"
config_name = "${config_name}"

try:
    with open("${CONFIG_MAP_FILE}", 'r') as f:
        config_map = json.load(f)
except:
    config_map = {}

config_map[config_file] = {
    "lora_dir": lora_dir,
    "config_name": config_name,
    "timestamp": __import__('datetime').datetime.now().isoformat()
}

with open("${CONFIG_MAP_FILE}", 'w') as f:
    json.dump(config_map, f, indent=2, ensure_ascii=False)

print(f"Saved mapping: {config_file} -> {lora_dir}")
EOF
}

# 从映射文件读取LoRA目录
get_lora_dir_from_map() {
    local config_file=$1
    if [ ! -f "${CONFIG_MAP_FILE}" ]; then
        echo ""
        return
    fi
    
    local lora_dir=$(python3 << EOF
import json
import sys

try:
    with open("${CONFIG_MAP_FILE}", 'r') as f:
        config_map = json.load(f)
    
    if "${config_file}" in config_map:
        print(config_map["${config_file}"]["lora_dir"])
    else:
        print("")
except:
    print("")
EOF
)
    echo "$lora_dir"
}

# 训练单个配置
train_config() {
    local config_file=$1
    local config_name=$2
    local config_path="./data/${config_file}"
    
    echo ""
    echo "=========================================="
    echo "训练配置: ${config_name}"
    echo "配置文件: ${config_file}"
    echo "=========================================="
    
    # 检查配置文件是否存在
    if [ ! -f "${config_path}" ]; then
        echo "错误: 配置文件不存在: ${config_path}"
        return 1
    fi
    
    # 记录训练前的LoRA目录（用于确定新创建的目录）
    local before_dirs=$(ls -td "${LORA_BASE_DIR}/${PREDICTION_MODEL}_"* 2>/dev/null | head -1)
    
    # 构建训练命令
    local train_cmd="python main.py \
        --dataset_path ${DATASET_PATH} \
        --model_name ${MODEL_NAME} \
        --prediction_model_name ${PREDICTION_MODEL} \
        --task train \
        --batch_size ${BATCH_SIZE} \
        --num_epochs ${NUM_EPOCHS} \
        --learning_rate ${LEARNING_RATE} \
        --lora_rank ${LORA_RANK} \
        --device ${DEVICE} \
        --top_k_values ${TOP_K_VALUES} \
        --corruption_index_path ${config_path}"
    
    # 执行训练
    echo "执行命令: ${train_cmd}"
    eval ${train_cmd}
    
    # 获取最新创建的LoRA目录
    local after_dirs=$(ls -td "${LORA_BASE_DIR}/${PREDICTION_MODEL}_"* 2>/dev/null | head -1)
    
    if [ -z "$after_dirs" ]; then
        echo "错误: 未找到新创建的LoRA目录"
        return 1
    fi
    
    # 如果before_dirs存在且与after_dirs相同，说明没有创建新目录
    if [ -n "$before_dirs" ] && [ "$before_dirs" == "$after_dirs" ]; then
        echo "警告: 可能没有创建新的LoRA目录，使用现有目录: ${after_dirs}"
    fi
    
    local lora_dir="${after_dirs}"
    echo "LoRA保存位置: ${lora_dir}"
    
    # 保存映射关系
    save_config_map "${config_file}" "${lora_dir}" "${config_name}"
    
    echo "训练完成: ${config_name}"
    echo ""
}

# 评估单个配置
eval_config() {
    local config_file=$1
    local config_name=$2
    local config_path="./data/${config_file}"
    
    echo ""
    echo "=========================================="
    echo "评估配置: ${config_name}"
    echo "配置文件: ${config_file}"
    echo "=========================================="
    
    # 从映射文件获取LoRA目录
    local lora_dir=$(get_lora_dir_from_map "${config_file}")
    
    if [ -z "$lora_dir" ]; then
        # 如果映射文件中没有，尝试获取最新的LoRA目录
        echo "警告: 映射文件中未找到LoRA目录，尝试使用最新目录..."
        lora_dir=$(get_latest_lora_dir "${PREDICTION_MODEL}")
    fi
    
    if [ -z "$lora_dir" ] || [ ! -d "$lora_dir" ]; then
        echo "错误: 未找到LoRA目录: ${lora_dir}"
        echo "请先运行训练模式或检查LoRA目录是否存在"
        return 1
    fi
    
    echo "LoRA目录: ${lora_dir}"
    
    # 构建评估命令
    local eval_cmd="python main.py \
        --dataset_path ${DATASET_PATH} \
        --model_name ${MODEL_NAME} \
        --prediction_model_name ${PREDICTION_MODEL} \
        --task eval \
        --batch_size ${BATCH_SIZE} \
        --device ${DEVICE} \
        --lora_load_dir ${lora_dir} \
        --eval_top_k_corruptions ${EVAL_TOP_K_CORRUPTIONS} \
        --top_k_values ${TOP_K_VALUES} \
        --corruption_index_path ${config_path}"
    
    # 执行评估
    echo "执行命令: ${eval_cmd}"
    eval ${eval_cmd}
    
    echo "评估完成: ${config_name}"
    echo ""
}

# ==================== 主函数 ====================

main() {
    # 解析命令行参数
    local mode_train=false
    local mode_eval=false
    
    if [ $# -eq 0 ]; then
        # 默认：训练+评估
        mode_train=true
        mode_eval=true
    else
        for arg in "$@"; do
            case "$arg" in
                train)
                    mode_train=true
                    ;;
                eval)
                    mode_eval=true
                    ;;
                *)
                    echo "未知参数: $arg"
                    echo "使用方法: $0 [train] [eval]"
                    echo "  不提供参数: 默认训练+评估"
                    echo "  train: 只训练"
                    echo "  eval: 只评估"
                    echo "  train eval: 训练后评估"
                    exit 1
                    ;;
            esac
        done
    fi
    
    # 创建必要目录
    setup_directories
    
    # 打印实验信息
    echo "=========================================="
    echo "Ablation实验：不同Corruption类型设置"
    echo "=========================================="
    echo "预测模型: ${PREDICTION_MODEL}"
    echo "数据集: ${DATASET_PATH}"
    echo "MLLM模型: ${MODEL_NAME}"
    echo "训练模式: ${mode_train}"
    echo "评估模式: ${mode_eval}"
    echo "配置文件数量: ${#CONFIG_FILES[@]}"
    echo ""
    
    # 训练阶段
    if [ "$mode_train" = true ]; then
        echo "开始训练阶段..."
        echo ""
        
        for config_entry in "${CONFIG_FILES[@]}"; do
            IFS=':' read -r config_file config_name <<< "$config_entry"
            train_config "$config_file" "$config_name"
        done
        
        echo "=========================================="
        echo "所有配置训练完成！"
        echo "=========================================="
        echo ""
        echo "配置映射文件: ${CONFIG_MAP_FILE}"
        echo "可以使用以下命令查看映射关系:"
        echo "  cat ${CONFIG_MAP_FILE}"
        echo ""
    fi
    
    # 评估阶段
    if [ "$mode_eval" = true ]; then
        echo "开始评估阶段..."
        echo ""
        
        # 如果只评估，检查映射文件是否存在
        if [ "$mode_train" = false ] && [ ! -f "${CONFIG_MAP_FILE}" ]; then
            echo "错误: 映射文件不存在: ${CONFIG_MAP_FILE}"
            echo "请先运行训练模式或手动创建映射文件"
            exit 1
        fi
        
        for config_entry in "${CONFIG_FILES[@]}"; do
            IFS=':' read -r config_file config_name <<< "$config_entry"
            eval_config "$config_file" "$config_name"
        done
        
        echo "=========================================="
        echo "所有配置评估完成！"
        echo "=========================================="
        echo ""
    fi
    
    echo "实验完成！"
}

# 运行主函数
main "$@"

