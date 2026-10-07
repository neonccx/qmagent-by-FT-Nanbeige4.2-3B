# QCal Agent Model

QCal Agent Model 是 [QCal Agent](https://github.com/neonccx/qcal-agent) 的
`v1.0.0` 单比特校准规划模型。它是基于
[Nanbeige/Nanbeige4.2-3B](https://huggingface.co/Nanbeige/Nanbeige4.2-3B)
训练的 PEFT LoRA adapter，不包含基座模型权重。

> 本模型只通过合成单比特校准数据与模拟闭环验收。它没有在真实量子处理器上完成
> 验证，也不包含耦合器、双比特门或自主真机操作能力。模型输出必须经过 QCal Agent
> 的受限动作协议与确定性控制器，不能直接作为仪器命令执行。

## 下载

从 [v1.0.0 Release](https://github.com/neonccx/qcal-agent-model/releases/tag/v1.0.0)
下载：

- `qcal-agent-1.0.0-adapter.tar.gz`：标准 PEFT adapter 包；
- `portable_manifest.json`：基座、tokenizer、数据划分、权重和评测哈希；
- `SHA256SUMS`：发布附件校验值。

模型包 SHA256：

```text
94b3433e74c663dbe79c6d28a261d913831198891a0e22e47d1b57b3aab69337
```

## 加载

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "Nanbeige/Nanbeige4.2-3B"
tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained(
    base_id,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)
model = PeftModel.from_pretrained(base, "qcal-agent-1.0.0-adapter")
```

## 验收结果

冻结 test 与 OOD 各 160 个样本：

| 指标 | Test | OOD |
|---|---:|---:|
| 原生调用有效率 | 100% | 100% |
| 下一工具准确率 | 100% | 100% |
| 有界参数完全准确率 | 98.75% | 100% |
| 控制器可执行率 | 100% | 100% |

新随机种子的模拟闭环为 3/3 accepted，且没有 invalid action、tool error 或
policy error。这些是合成数据指标，不代表真机性能。

## 仓库分工

- 本仓库：模型卡、评测清单与 LoRA 发布附件；
- [`qcal-agent`](https://github.com/neonccx/qcal-agent)：Agent、测控模拟、数据生成、
  训练、评测和真机适配接口。

## 许可证

本仓库使用 Apache License 2.0。Nanbeige 基座权重不包含在本仓库中，使用时还应
遵守其上游模型许可和模型卡说明。
