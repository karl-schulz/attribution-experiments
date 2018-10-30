
import torch
import torch.nn as nn
from mlproject.model import Model

class ModelHolder(Model):

    def __init__(self, model, optimizer, loss, device):
        super().__init__(device)
        self.model = model  # self.build_model()
        self.optimizer = optimizer
        self.loss = loss

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.model(*args, **kwargs)

    def get_model(self) -> nn.Module:
        return self.model

    @staticmethod
    def accuracy(labels: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        return torch.mean(scores.argmax(dim=1).eq(labels).float())

    def test_batch(self, batch) -> dict:
        input, labels = self.to_device(batch)
        out = self(input)
        loss = self.loss(out, labels)
        acc = self.accuracy(labels=labels, scores=out)
        return {'loss': loss, 'accuracy': acc}

    def train_batch(self, batch) -> dict:
        input, labels = self.to_device(batch)
        self.optimizer.zero_grad()
        output = self(input)
        loss = self.loss(output, labels)
        loss.backward()
        self.optimizer.step()
        return {'loss': loss}
