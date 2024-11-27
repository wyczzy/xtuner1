# Copyright (c) OpenMMLab. All rights reserved.
# <<<<<<< HEAD
# from .default_collate_fn import default_collate_fn
# =======
from .default_collate_fn import anyshape_llava_collate_fn, default_collate_fn
# >>>>>>> pr-460
from .mmlu_collate_fn import mmlu_collate_fn

__all__ = [
    'default_collate_fn', 'mmlu_collate_fn', 'anyshape_llava_collate_fn'
]
