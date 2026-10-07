# QCal Agent

QCal Agent 是面向超导量子单比特自动校准的完整第一版工程。一个仓库同时包含：

- 可审计的 Agent 状态机、严格工具协议、预算和回滚；
- 可由物理模拟器切换到注册实验室适配器的采集与分析层；
- S21、Power2D、ZPA2D、能谱、Rabi、Ramsey、T1、Echo、IQraw 和单比特 XEB；
- 可加载 Nanbeige4.2-3B 与 LoRA 的本地或远程模型推理层；
- 内容寻址原始数据、拟合质量门控和留出 IQ shot 验收。

正式训练使用的代码、数据、基座权重和 LoRA 已从运行仓库拆分到同级的
`training-Nanbeige4.2-3B/`。普通 Agent 用户不需要下载该目录。

本仓库已发布 `v1.0.0`，不要求用户理解历史实验版本。新 LoRA 已完成训练并通过配置的模拟数据发布门禁，便携 PEFT 包已在服务器打包和加载验证。模型卡、评测清单和下载附件独立发布在 [`qcal-agent-model`](https://github.com/neonccx/qcal-agent-model) 的 [v1.0.0 Release](https://github.com/neonccx/qcal-agent-model/releases/tag/v1.0.0)。代码版本号和模拟门禁均不代表已通过真机验收。旧数据和旧模型只用于内部对照，不进入第一版命名或默认配置。

## 安全与科研边界

默认运行的是物理启发模拟器，不会连接真实仪器。语言模型只选择注册实验和有界参数，不拟合原始数组、不执行任意 Python/Shell，也不能绕过控制器判定成功。Agent 真机接入通过 `RegisteredHardwareBackend` 显式注册每个采集函数，并要求逐次操作确认；部署时仍须补齐设备通道映射、硬件限幅、斜率限制、急停、独占锁和监督干跑证据。

当前单比特 XEB 是保真度衰减代理实验，不能当作 Google 多比特随机线路采样或真机门保真度证据。IQraw 读出保真度使用留出 shot 计算；名义 |0⟩ 误判比例不能单独推断热布居或有效温度。当前没有实现耦合器、`generate_coupler` 或双比特门；ZPA2D 是单比特 Z/磁通偏置与读出频率二维扫描，不依赖耦合器。

当前可行性与待补证据见 [项目评估](docs/FEASIBILITY_REVIEW.md)。

## 本地运行

```bash
conda env create -f environment.yml -p ./.conda-env
conda activate ./.conda-env
python -m pip install -e . --no-build-isolation
pytest -q
qm-agent run --policy rule --backend physical --output-dir runs/rule-demo
```

如需训练或复现实验，必须同时下载本仓库和同级的
`training-Nanbeige4.2-3B/`；训练代码会直接使用本仓库的 `src/qmagent`。

## 目录

- `src/qmagent/`：Agent、协议、控制器、物理模拟和报告。
- `tests/`：Agent 与测控运行时测试。
- `docs/`：第一版架构、真机接入和评测声明。

## 本地 Agent 连接模型服务器

远程模式在本地运行 Agent、控制器、后端、会话和报告；服务器只加载模型并响应 `decide`/`chat`：

```bash
qm-agent remote \
  --host bishe-5090 \
  --project /home/caochuangxin/bishe/qcal-agent \
  --control-path ~/.ssh/qcal-agent-5090.sock \
  --server-home /home/caochuangxin/bishe/qcal-agent-home
qm-agent connect
```

校准完成后执行 `/report`。会话状态、实验数据与报告始终保存在本地
`--home` 下；服务器端 `home` 仅提供 `policy=hf` 的模型配置。旧配置中的
`local_results` 字段仍可读取，但不再使用。
