import torch.optim as optim
import torch.nn as nn

def GetCorrectPredCount(pPrediction, pLabels):
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

class TrainConfig:
    def __init__(self, model, num_epochs = 20):
        self.optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=15, gamma=0.1, verbose=True)
        self.criterion = nn.CrossEntropyLoss()
        self.num_epochs = num_epochs
