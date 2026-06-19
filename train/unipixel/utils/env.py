# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import torch


def get_auto_device():
    try:
        import torch_npu
        has_npu = torch_npu.npu.is_available()
    except ImportError:
        has_npu = False

    return 'cuda' if torch.cuda.is_available() else 'npu' if has_npu else 'cpu'
