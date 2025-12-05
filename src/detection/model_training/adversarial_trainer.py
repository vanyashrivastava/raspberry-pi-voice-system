# Owner: Samya
# Responsibility: Train main classifier with an adversarial debiasing head.
# Dependencies: torch, sklearn, numpy

from __future__ import annotations
import typing as t

import torch
import torch.nn as nn
import torch.optim as optim


class GradientReversal(torch.autograd.Function):
    """
    Classic gradient reversal layer used in adversarial debiasing / domain adaptation.
    Forward: identity
    Backward: multiplies gradient by -lambda_
    """

    @staticmethod
    def forward(ctx, x, lambda_: float):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x, lambda_: float):
    return GradientReversal.apply(x, lambda_)


class AdversarialDebiasingModel(nn.Module):
    """
    Wraps a base encoder with:
      - main head: predicts y (scam / not scam)
      - adversary head: predicts sensitive attribute z (e.g., locale, user group)
    """

    def __init__(
        self,
        encoder: nn.Module,
        hidden_dim: int,
        num_labels: int = 2,
        num_sensitive: int = 2,
    ):
        super().__init__()
        self.encoder = encoder  # e.g., a pooled HF encoder or any text encoder

        self.classifier = nn.Linear(hidden_dim, num_labels)
        self.adversary = nn.Linear(hidden_dim, num_sensitive)

    def forward(self, features, lambda_adv: float = 0.0):
        """
        features: representation from encoder (batch_size, hidden_dim)
        lambda_adv: strength of gradient reversal
        """
        # main prediction
        logits_main = self.classifier(features)

        # adversary prediction with gradient reversal
        rev = grad_reverse(features, lambda_adv)
        logits_adv = self.adversary(rev)

        return logits_main, logits_adv


class AdversarialTrainer:
    """
    Trainer for adversarial debiasing.

    Usage:
        trainer = AdversarialTrainer(model, lambda_adv=0.5, lr=1e-4)
        trainer.train(dataloader, num_epochs=3, device="cuda")

    Assumes each batch from dataloader yields:
        input_ids, attention_mask, labels_y, labels_sensitive
    """

    def __init__(
        self,
        encoder: nn.Module,
        hidden_dim: int,
        num_labels: int = 2,
        num_sensitive: int = 2,
        lambda_adv: float = 0.5,
        lr: float = 1e-4,
    ):
        self.lambda_adv = lambda_adv

        self.model = AdversarialDebiasingModel(
            encoder=encoder,
            hidden_dim=hidden_dim,
            num_labels=num_labels,
            num_sensitive=num_sensitive,
        )

        self.criterion_main = nn.CrossEntropyLoss()
        self.criterion_adv = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, dataloader, num_epochs: int = 3, device: str = "cpu"):
        self.model.to(device)
        self.model.train()

        for epoch in range(num_epochs):
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                y = batch["labels"].to(device)              # scam / not scam
                z = batch["sensitive_attr"].to(device)      # group / protected attr

                # ---- 1) Encode text (using your encoder) ----
                # Example for HF encoder: encoder returns last_hidden_state + pooler
                outputs = self.model.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                # Choose pooled representation; adjust depending on your encoder
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    features = outputs.pooler_output           # (batch, hidden_dim)
                else:
                    # fall back to averaging last_hidden_state
                    features = outputs.last_hidden_state.mean(dim=1)

                # ---- 2) Main + adversary heads ----
                logits_main, logits_adv = self.model(features, lambda_adv=self.lambda_adv)

                # ---- 3) Compute losses ----
                loss_main = self.criterion_main(logits_main, y)
                loss_adv = self.criterion_adv(logits_adv, z)

                # Combined objective:
                #   Minimize main loss, maximize adversary loss
                #   Gradient reversal handles the "maximize", so we ADD them:
                loss = loss_main + loss_adv

                # ---- 4) Backprop + step ----
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # You can log these if you want
            print(f"[epoch {epoch+1}/{num_epochs}] loss_main={loss_main.item():.4f} loss_adv={loss_adv.item():.4f}")

        return self.model

if __name__ == "__main__":
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModel

    # -------------------------------
    # 1. Tiny dataset wrapper
    # -------------------------------
    class ScamDataset(Dataset):
        """
        Simple dataset that:
          - tokenizes text with a HF tokenizer
          - returns dict with: input_ids, attention_mask, labels, sensitive_attr
        """

        def __init__(self, texts, labels, sensitive_attrs, tokenizer, max_length: int = 128):
            self.texts = texts
            self.labels = labels
            self.sensitive_attrs = sensitive_attrs
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            label = self.labels[idx]
            sens = self.sensitive_attrs[idx]

            enc = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            item = {
                "input_ids": enc["input_ids"].squeeze(0),        # (seq_len,)
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": torch.tensor(label, dtype=torch.long),
                "sensitive_attr": torch.tensor(sens, dtype=torch.long),
            }
            return item

    # -------------------------------
    # 2. Load encoder + tokenizer
    # -------------------------------
    model_name = "distilbert-base-uncased"  # you can change this
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name)

    hidden_dim = encoder.config.hidden_size  # e.g. 768 for DistilBERT

    # -------------------------------
    # 3. Dummy data (replace with real data later)
    # -------------------------------
    texts = [
        "You have won a free iPhone! Click here to claim your prize now.",
        "Hi, just checking in about our meeting tomorrow at 3 PM.",
        "Urgent: Your bank account has been compromised, reset your password now.",
        "Reminder: your package will be delivered today between 2 and 4 PM.",
    ]
    # 1 = scam, 0 = not scam
    labels = [1, 0, 1, 0]

    # Example sensitive attribute: which user group / locale / channel, etc.
    # Here just dummy 0/1 groups.
    sensitive_attrs = [0, 1, 0, 1]

    dataset = ScamDataset(texts, labels, sensitive_attrs, tokenizer, max_length=128)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # -------------------------------
    # 4. Create trainer + train
    # -------------------------------
    trainer = AdversarialTrainer(
        encoder=encoder,
        hidden_dim=hidden_dim,
        num_labels=2,
        num_sensitive=2,
        lambda_adv=0.5,
        lr=1e-4,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    trained_model = trainer.train(
        dataloader=dataloader,
        num_epochs=3,
        device=device,
    )

    print("Training finished.")
