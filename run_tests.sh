#!/bin/bash

echo "Running unit tests..."

# 确保Python路径包含当前目录
export PYTHONPATH=".:${PYTHONPATH}"

# 运行所有测试
python -m unittest discover -s tests -p "test_*.py" -v

# 检查测试结果
if [ $? -eq 0 ]; then
    echo "All tests passed successfully!"
    exit 0
else
    echo "Some tests failed!"
    exit 1
fi