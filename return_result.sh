#!/bin/bash

# 设置工号（实际使用时需要修改）
CONTESTANT_ID="12345"

# 目标目录路径（根据实际情况修改）
TARGET_DIR="/path/to/your/personal/submission/directory"

# 查找最新的结果文件
RESULT_FILE=$(ls -t ${CONTESTANT_ID}_*.csv | head -n 1)

if [ -z "$RESULT_FILE" ]; then
    echo "Error: No result file found for ${CONTESTANT_ID}."
    exit 1
fi

echo "Copying '$RESULT_FILE' to '$TARGET_DIR'..."
cp "$RESULT_FILE" "$TARGET_DIR/"

if [ $? -eq 0 ]; then
    echo "Copy successful."
    exit 0
else
    echo "Error: Copy failed."
    exit 1
fi