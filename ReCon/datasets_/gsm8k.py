import re
import torch
from datasets import load_dataset

GSM_SYSTEM_PROMPT = """You are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{}. 
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>"""


def extract_answer_gsm8k(text):
    """Numeric answer after '####' in the GSM8K reference solution."""
    m = re.search(r"####\s*\$?([\d,]+(?:\.\d+)?)", text)
    return float(m.group(1).replace(",", "")) if m else None


class GSM8KDataset(torch.utils.data.Dataset):
    """Base class: zero-shot chat prompt = system prompt + question, chat template, then '<reasoning>'.
    Items are (prompt, question, answer)."""
    system_prompt = GSM_SYSTEM_PROMPT

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.load_test_dataset()

    def load_test_dataset(self):
        self.dataset = load_dataset("gsm8k", "main", split="test")

    def __len__(self):
        return len(self.dataset)

    def create_prompt(self, question):
        messages = [{"role": "user", "content": self.system_prompt + "\n\n" + question}]
        return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False) + "<reasoning>"

    def __getitem__(self, idx):
        row = self.dataset[idx]
        return self.create_prompt(row["question"]), row["question"], extract_answer_gsm8k(row["answer"])

    def collate_fn(self, batch):
        prompts, questions, answers = zip(*batch)
        input_ids = self.tokenizer(list(prompts), padding_side="left", return_tensors="pt", padding="longest").input_ids
        return {"input_ids": input_ids, "questions": list(questions), "answers": list(answers), "prompts": list(prompts)}
