from datasets import load_dataset
from .gsm8k import GSM8KDataset

MATH500_SYSTEM_PROMPT = """You are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{}.
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>" 
"""


class MATH500Dataset(GSM8KDataset):
    """MATH-500 (HuggingFaceH4/MATH-500 test split); the answer is the reference answer string."""
    system_prompt = MATH500_SYSTEM_PROMPT

    def load_test_dataset(self):
        self.dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

    def __getitem__(self, idx):
        row = self.dataset[idx]
        return self.create_prompt(row["problem"]), row["problem"], row["answer"]
