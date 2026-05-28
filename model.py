import torch.nn as nn
import torch.nn.functional as F

class MorseCRNN_v2(nn.Module):
    def __init__(self, num_classes, n_mels=80, hidden_size=256):
        super().__init__()
        self.conv1a = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1a = nn.BatchNorm2d(32)
        self.conv1b = nn.Conv2d(32, 32, 3, padding=1)
        self.bn1b = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(0.1)

        self.conv2a = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2a = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2b = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout2d(0.1)

        self.conv3a = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3a = nn.BatchNorm2d(128)
        self.conv3b = nn.Conv2d(128, 128, 3, padding=1)
        self.bn3b = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout2d(0.1)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))
        self.rnn = nn.LSTM(128, hidden_size, num_layers=3, bidirectional=True,
                           dropout=0.3, batch_first=False)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1).permute(0, 1, 3, 2)
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = self.pool1(x); x = self.drop1(x)
        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = self.pool2(x); x = self.drop2(x)
        x = F.relu(self.bn3a(self.conv3a(x)))
        x = F.relu(self.bn3b(self.conv3b(x)))
        x = self.pool3(x); x = self.drop3(x)
        x = self.adaptive_pool(x)
        x = x.squeeze(2).permute(2,0,1)
        x, _ = self.rnn(x)
        logits = self.classifier(x)
        return logits