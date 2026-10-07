import os

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    device = torch.device("cuda", local_rank)
    tensor = torch.tensor([float(local_rank + 1)], device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    expected = dist.get_world_size() * (dist.get_world_size() + 1) / 2
    if tensor.item() != expected:
        raise RuntimeError(f"NCCL all-reduce failed: {tensor.item()} != {expected}")

    print(
        f"rank={dist.get_rank()} device={torch.cuda.get_device_name(local_rank)} "
        f"all_reduce={tensor.item():.1f} ok"
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
