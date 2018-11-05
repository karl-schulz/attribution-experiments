
import torch
import torch.nn as nn
from mlproject.model import Model

class Setup(Model):
    # TODO loss in eigene klasse auslagern
    def __init__(self, config):
        super().__init__(config["device"])
        self.model = self.build_model(config)
        self.optimizer = self.build_optimizer(config)
        self.loss = self.build_loss(config)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.model(*args, **kwargs)

    def test_batch(self, batch) -> dict:
        inputs, labels = self.to_device(batch)
        loss = self.loss.get_loss(inputs, labels)
        out = self.forward(inputs)
        acc = torch.mean(out.argmax(dim=1).eq(labels).float())
        return {'loss': loss, 'accuracy': acc}

    def train_batch(self, batch) -> dict:
        self.optimizer.zero_grad()
        loss = self.loss.get_loss(*self.to_device(batch))
        loss.backward()
        self.optimizer.step()
        return {'loss': loss}

    def build_loss(self, config):
        raise NotImplementedError

    def build_model(self, config):
        raise NotImplementedError

    def build_optimizer(self, config):
        raise NotImplementedError
