"""GSM8K / ARC-C / TruthfulQA-MC / MATH-500 (HF datasets; downloaded on first use, cached afterwards)."""
from .gsm8k import GSM8KDataset
from .arcc import ARCCDataset
from .truthfulqa import TruthfulQAMCDataset
from .math500 import MATH500Dataset

DATASET_MAP = {"gsm8k": GSM8KDataset, "arcc": ARCCDataset, "truthfulqa": TruthfulQAMCDataset, "math": MATH500Dataset}
