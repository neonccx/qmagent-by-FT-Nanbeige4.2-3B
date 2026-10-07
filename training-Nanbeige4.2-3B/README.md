# Nanbeige4.2-3B training workspace for QCal Agent

本目录保存 QCal Agent 的训练与实验复现材料，不属于 Agent 的运行时源码。

## 必须同时下载两个文件夹

训练代码会直接导入 `qcal-agent/src/qmagent` 中的协议、控制器、模拟器和
Policy 实现。请把两个目录放在同一级，名称保持如下：

```text
workspace/
├── qcal-agent/
└── training-Nanbeige4.2-3B/
```

只下载本目录无法训练；脚本会以明确错误退出。只使用已训练 Agent 时只需要
`qcal-agent`，无需下载本训练目录和基座模型。

## 目录

- `training/`：数据生成、审计、LoRA 训练、评估和发布脚本。
- `env.sh`：选择服务器训练环境并设置 Hugging Face/PyTorch 缓存。
- `cache/`：训练和模型推理使用的 Hugging Face/PyTorch 本地缓存。
- `tests/`：训练与评估工具测试。
- `gpu_smoke_test.py`：训练前检查多 GPU、CUDA 和 NCCL all-reduce。
- `models/Nanbeige4.2-3B/`：本地基座模型权重（约 7.8 GB）。
- `models/qcal-agent-1.0.0-adapter/`：训练完成的 LoRA 权重与验收清单。
- `datasets/`：训练、验证、冻结测试和 OOD 数据。
- `releases/`：数据集与 LoRA 发布包和校验文件。
- `qcal-agent-model/`：发布模型卡仓库镜像。
- `docs/`：数据集、训练选择和模型发布说明。

## 使用

```bash
cd training-Nanbeige4.2-3B
. ./env.sh
export PYTHONPATH="../qcal-agent/src:$PWD/training"

python training/audit_dataset.py datasets/qcal-agent-v1.0.0
python training/audit_tokens.py \
  --dataset datasets/qcal-agent-v1.0.0 \
  --model models/Nanbeige4.2-3B

python training/run_training_pipeline.py \
  --model models/Nanbeige4.2-3B \
  --dataset datasets/qcal-agent-v1.0.0 \
  --run-dir runs/new-run \
  --gpu 0
```

新 checkpoint、最终 LoRA 和评估结果应保存在本目录的 `runs/` 下。

四卡训练前可以先检查 GPU 通信：

```bash
torchrun --standalone --nproc_per_node=4 gpu_smoke_test.py
```
