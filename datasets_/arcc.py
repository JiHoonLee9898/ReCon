from datasets import load_dataset
from .gsm8k import GSM8KDataset

ARC_C_SYSTEM_PROMPT = """You are a science and reasoning expert. You will be given a multiple-choice question with options labeled A, B, C, D (and sometimes E).
Think step by step, but output only ONE final choice.
Wrap ONLY the final choice letter in a \\boxed{}.

Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>"""


class ARCCDataset(GSM8KDataset):
    """ARC-Challenge test split; the answer is the option letter."""
    system_prompt = ARC_C_SYSTEM_PROMPT

    def load_test_dataset(self):
        self.dataset = load_dataset("ai2_arc", "ARC-Challenge", split="test")

    @staticmethod
    def format_question(row):
        opts = "\n".join(f"{str(lab).strip()}. {str(txt).strip()}"
                         for lab, txt in zip(row["choices"]["label"], row["choices"]["text"]))
        return f"{str(row['question']).strip()}\n\nOptions:\n{opts}\n\nChoose ONE letter."

    def __getitem__(self, idx):
        row = self.dataset[idx]
        question = self.format_question(row)
        return self.create_prompt(question), question, str(row["answerKey"]).strip()
