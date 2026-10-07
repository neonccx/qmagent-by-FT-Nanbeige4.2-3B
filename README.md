# qmagent by FT Nanbeige4.2-3B

这是一个面向超导量子单比特校准的受约束 Agent，以及基于
[Nanbeige4.2-3B](https://huggingface.co/Nanbeige/Nanbeige4.2-3B) 的 LoRA
微调复现工作区。仓库将运行时与训练材料放在一起，但保持清晰边界：

```text
qcal-agent/                    Agent、控制器、模拟/硬件后端和报告
training-Nanbeige4.2-3B/      数据生成、训练、评估、模型元数据和发布证据
qcal-agent-home/              可移植的服务器模型配置模板
download_models.py            下载上游基座模型和本项目 LoRA
download_workspace.py         下载完整毕设工作区快照
restore_workspace.sh          校验并还原完整快照
```

## 完整复现：还原与原服务器相同的工作区

Git 仓库适合浏览和修改代码，但 GitHub 不允许普通 Git 文件超过 100 MB，而本项目
的基座模型分片分别约为 5.0 GB 和 3.4 GB。因此 `v1.0.0` Release 另附完整
`bishe` 工作区的分卷快照。它包含发布时服务器上的：

- `qcal-agent/`；
- `training-Nanbeige4.2-3B/`，包括基座权重、LoRA 权重、数据集、缓存和发布证据；
- `qcal-agent-home/`，包括当时的配置、会话与报告；
- 根目录 `README.md`。

快照只排除论文目录 `thesis/` 和 Git 内部元数据 `.git/`。下载后会逐卷校验
SHA256，再恢复为名为 `bishe/` 的目录：

```bash
git clone https://github.com/neonccx/qmagent-by-FT-Nanbeige4.2-3B.git
cd qmagent-by-FT-Nanbeige4.2-3B
python download_workspace.py
./restore_workspace.sh ..
```

文件级 `FILE_MANIFEST.sha256` 可验证恢复结果。快照保留了原服务器的绝对路径配置；
如果新服务器用户名或目录不同，请在恢复后修改 `bishe/qcal-agent-home/config.json`
中的模型路径，并重新安装环境。`qcal-agent/qm-agent` 是原服务器环境生成的入口
链接，换机器后应运行：

```bash
cd ../bishe/qcal-agent
ln -sfn "$(command -v qm-agent)" qm-agent
```

## 重要边界

- 默认后端是模拟器，不代表真实量子硬件验收。
- 模型只提出注册动作；参数范围、前置条件、预算和成功判定由确定性控制器负责。
- `model-rpc` 只提供 `info`、`decide`、`chat` 和 `close`，不执行实验或 Shell。
- Nanbeige 基座权重体积超过 GitHub 限制，遵循上游 Apache-2.0 发布并从
  Hugging Face 下载。本仓库不重复提交约 8 GB 基座权重。
- LoRA 权重作为本仓库 `v1.0.0` Release 附件发布，SHA256 为
  `94b3433e74c663dbe79c6d28a261d913831198891a0e22e47d1b57b3aab69337`。

## A. 只在本地使用 Agent（不使用大模型）

只需要 `qcal-agent/`，无需下载基座模型：

```bash
cd qcal-agent
conda env create -f environment.yml -p ./.conda-env
conda activate ./.conda-env
python -m pip install -e '.[terminal,report]' --no-build-isolation
qm-agent --home ~/.qm-agent-home shell --policy rule --backend physical
```

Rule 模式使用确定性策略，可运行完整模拟校准，但不能回答开放式聊天问题。
会话和报告保存在 `~/.qm-agent-home`，不会上传到服务器。

## B. 在自己的服务器部署微调模型

服务器需要 NVIDIA GPU、SSH、PyTorch，并同时保留仓库中的三个顶层目录。

### 1. 创建服务器环境并下载模型

```bash
cd training-Nanbeige4.2-3B
conda env create -f environment.yml -p ./.conda-env
conda activate ./.conda-env
cd ..
python download_models.py
python -m pip install -e './qcal-agent[model]' --no-build-isolation
```

### 2. 建立服务器配置

```bash
cp qcal-agent-home/config.example.json qcal-agent-home/config.json
```

编辑 `config.json`，把 `/ABSOLUTE/PATH/TO/REPOSITORY` 替换为仓库绝对路径。
验证模型端点（该命令本身不加载权重）：

```bash
printf '%s\n' \
  '{"id":1,"method":"info","params":{}}' \
  '{"id":2,"method":"close","params":{}}' |
  qm-agent model-rpc --home "$PWD/qcal-agent-home"
```

### 3. 从另一台电脑连接

客户端只安装 `qcal-agent`，并先配置免交互 SSH 密钥。然后执行：

```bash
qm-agent --home ~/.qm-agent-home remote \
  --host YOUR_SSH_HOST \
  --project /ABSOLUTE/PATH/TO/REPOSITORY/qcal-agent \
  --control-path /tmp/qmagent-model.sock \
  --server-home /ABSOLUTE/PATH/TO/REPOSITORY/qcal-agent-home

qm-agent --home ~/.qm-agent-home connect
```

本地电脑运行 Agent、控制器、实验后端、会话和报告；SSH 服务器只运行模型推理。

## C. 从头复现 LoRA 微调

训练代码直接导入相邻的 `qcal-agent/src/qmagent`。不要单独复制训练脚本，否则
训练协议可能与运行时不一致。

```bash
cd training-Nanbeige4.2-3B
conda env create -f environment.yml -p ./.conda-env
conda activate ./.conda-env
. ./env.sh
export PYTHONPATH="../qcal-agent/src:$PWD/training"
```

仓库中的已发布 JSONL 分割可核验冻结评估数据，但公开压缩包不含每条记录引用的
全部原始模拟 artifact。因此，完整的 fail-closed 复现应重新生成数据：

```bash
python training/build_dataset.py \
  --devices 64 \
  --output datasets/rebuilt-v1

python training/audit_dataset.py datasets/rebuilt-v1
python training/audit_tokens.py \
  --dataset datasets/rebuilt-v1 \
  --model models/Nanbeige4.2-3B
```

启动完整基线、LoRA、冻结测试/OOD、闭环评估和发布门禁：

```bash
python training/run_training_pipeline.py \
  --model models/Nanbeige4.2-3B \
  --dataset datasets/rebuilt-v1 \
  --run-dir runs/reproduction \
  --gpu 0
```

运行会保存配置、源码哈希、预测、控制器评分、checkpoint、最终 adapter 和发布
门禁结果。训练中断时必须使用流水线提供的显式 checkpoint 恢复参数，不能覆盖
原始证据。详细数据与发布边界见 `training-Nanbeige4.2-3B/docs/`。

## 测试

```bash
cd qcal-agent
pytest -q

cd ../training-Nanbeige4.2-3B
export PYTHONPATH="../qcal-agent/src:$PWD:$PWD/training"
pytest -q tests
```

## 许可证

Agent 代码见 `qcal-agent/LICENSE`。Nanbeige 基座模型许可证与模型卡见其
Hugging Face 页面；LoRA 模型卡与限制见
`training-Nanbeige4.2-3B/qcal-agent-model/README.md`。
