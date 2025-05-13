#!/bin/bash

# 设置工号（实际使用时需要修改）
WORKER_ID="12345"

echo "Setting up environment..."
# 激活conda环境（根据实际环境调整）
# source /opt/conda/bin/activate your_env_name

# 如果使用打包的环境：
# tar -xzf aitmg_2025_${WORKER_ID}.tar.gz -C /tmp/
# source /tmp/aitmg_2025_${WORKER_ID}/bin/activate

# 安装依赖
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Starting main inference process..."
# 确保示例数据可访问
python main.py \
    --labeled-data /home/jovyan/demo_data/labeled_demo_data_aitmg_202504.csv \
    --unlabeled-data /home/jovyan/demo_data/unlabeled_demo_data_aitmg_202504.csv \
    --output-dir . \
    --config config.json

# 检查输出
echo "Inference finished. Checking output..."
OUTPUT_FILE=$(ls -t ${WORKER_ID}_*.csv | head -n 1)
if [ -f "$OUTPUT_FILE" ] && [ -s "$OUTPUT_FILE" ]; then
    echo "Output file '$OUTPUT_FILE' created successfully."
    exit 0
else
    echo "Error: Output file not found or empty!"
    exit 1
fi