# QM Agent

QM Agent 用于超导量子单比特的自动校准。它让策略模型选择实验，由确定性控制器
负责参数范围、实验顺序、预算、状态更新和成功判定。

当前版本支持 S21、读出功率扫描、Z 偏置扫描、能谱、Rabi、Ramsey、T1、Echo、
IQ 读出和单比特 XEB。你可以先用内置物理模拟器运行完整流程，也可以通过注册
硬件采集函数接入实验设备。

## 选择使用方式

| 目标 | 需要的内容 |
| --- | --- |
| 在一台电脑上体验校准流程 | `qcal-agent/`，使用规则策略和模拟器 |
| 本地控制实验，服务器运行模型 | `qcal-agent/`、模型服务器和 LoRA 权重 |
| 复现数据生成、微调和评估 | `qcal-agent/` 与 `training-Nanbeige4.2-3B/` |
| 还原发布时的完整运行环境 | `v1.0.0` Release 中的工作区快照 |

## 本地运行

本地模式不需要 GPU，也不需要下载大模型：

```bash
git clone https://github.com/neonccx/qmagent-by-FT-Nanbeige4.2-3B.git
cd qmagent-by-FT-Nanbeige4.2-3B/qcal-agent

conda env create -f environment.yml -p ./.conda-env
conda activate ./.conda-env
python -m pip install -e '.[terminal,report]' --no-build-isolation

qm-agent --home ~/.qm-agent-home shell --policy rule --backend physical
```

规则策略可以运行校准流程，但不处理开放式对话。会话、原始记录和报告保存在
`~/.qm-agent-home`。

## 部署模型服务器

模型服务器需要 NVIDIA GPU、SSH 和能够运行 PyTorch 的 Linux 环境。

```bash
git clone https://github.com/neonccx/qmagent-by-FT-Nanbeige4.2-3B.git
cd qmagent-by-FT-Nanbeige4.2-3B/training-Nanbeige4.2-3B

conda env create -f environment.yml -p ./.conda-env
conda activate ./.conda-env
cd ..

python download_models.py
python -m pip install -e './qcal-agent[model]' --no-build-isolation
cp qcal-agent-home/config.example.json qcal-agent-home/config.json
```

编辑 `qcal-agent-home/config.json`，把示例中的仓库路径换成服务器上的绝对路径。
随后可以用一条请求检查模型服务入口：

```bash
printf '%s\n' \
  '{"id":1,"method":"info","params":{}}' \
  '{"id":2,"method":"close","params":{}}' |
  qm-agent model-rpc --home "$PWD/qcal-agent-home"
```

`model-rpc` 只接收模型信息、决策、对话和关闭请求。实验执行、硬件访问、会话和
报告都留在客户端电脑上。

## 从客户端连接模型服务器

先在客户端安装 `qcal-agent`，并配置可以免交互登录服务器的 SSH 密钥：

```bash
qm-agent --home ~/.qm-agent-home remote \
  --host YOUR_SSH_HOST \
  --project /opt/qmagent/qcal-agent \
  --control-path ~/.ssh/qmagent-model.sock \
  --server-home /opt/qmagent/qcal-agent-home

qm-agent --home ~/.qm-agent-home connect
```

服务器上的目录可以自行选择，只要 `--project` 和 `--server-home` 与实际位置一致。

## 复现 LoRA 微调

训练脚本会直接导入相邻目录中的 `qcal-agent/src/qmagent`。克隆仓库后请保留两个
目录的相对位置：

```text
qmagent-by-FT-Nanbeige4.2-3B/
├── qcal-agent/
└── training-Nanbeige4.2-3B/
```

准备环境：

```bash
cd training-Nanbeige4.2-3B
conda env create -f environment.yml -p ./.conda-env
conda activate ./.conda-env
. ./env.sh
export PYTHONPATH="../qcal-agent/src:$PWD/training"
```

重新生成并审计数据：

```bash
python training/build_dataset.py --devices 64 --output datasets/rebuilt-v1
python training/audit_dataset.py datasets/rebuilt-v1
python training/audit_tokens.py \
  --dataset datasets/rebuilt-v1 \
  --model models/Nanbeige4.2-3B
```

运行基线评估、LoRA 微调、冻结测试、OOD 评估和发布检查：

```bash
python training/run_training_pipeline.py \
  --model models/Nanbeige4.2-3B \
  --dataset datasets/rebuilt-v1 \
  --run-dir runs/reproduction \
  --gpu 0
```

训练输出包含配置、源码哈希、预测结果、控制器评分、checkpoint、最终 adapter 和
发布检查结果。数据格式和发布条件见
[`training-Nanbeige4.2-3B/docs/`](training-Nanbeige4.2-3B/docs/)。

## 下载完整工作区

基座模型文件超过 GitHub 的普通文件大小限制，因此完整工作区放在
[`v1.0.0` Release](https://github.com/neonccx/qmagent-by-FT-Nanbeige4.2-3B/releases/tag/v1.0.0)
中。下载脚本会取得四个分卷并检查 SHA256：

```bash
python download_workspace.py
./restore_workspace.sh ..
```

恢复结果位于 `../qmagent-workspace/`。其中包含 Agent、训练代码、基座模型、LoRA、
数据集、发布记录、会话和报告。换一台服务器后，需要修改
`qmagent-workspace/qcal-agent-home/config.json` 中的绝对路径，并重新创建命令入口：

```bash
cd ../qmagent-workspace/qcal-agent
ln -sfn "$(command -v qm-agent)" qm-agent
```

## 项目结构

```text
qcal-agent/                    Agent、控制器、后端和报告
training-Nanbeige4.2-3B/      数据、训练、评估和模型发布工具
qcal-agent-home/              模型服务器配置模板
download_models.py            下载基座模型和 LoRA
download_workspace.py         下载完整工作区
restore_workspace.sh          校验并恢复完整工作区
```

## 当前范围

默认后端是模拟器。模拟结果不能作为真实量子硬件的验收结论。接入真机时，还需要
完成通道映射、硬件限幅、斜率限制、设备锁、超时、急停和监督运行验证。

当前 XEB 是单比特保真度衰减代理实验。项目尚未实现耦合器控制和双比特门校准。

## 许可证

Agent 代码使用 [`qcal-agent/LICENSE`](qcal-agent/LICENSE)。Nanbeige 基座模型遵循
其上游模型卡与许可证；LoRA 的说明和适用范围见
[`training-Nanbeige4.2-3B/qcal-agent-model/README.md`](training-Nanbeige4.2-3B/qcal-agent-model/README.md)。
