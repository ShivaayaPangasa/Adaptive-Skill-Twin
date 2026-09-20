import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_size=117, hidden_size=128, num_layers=2, num_classes=4):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                              batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)       # h_n: (num_layers, batch, hidden)
        last_hidden = h_n[-1]             # final layer's hidden state
        return self.fc(last_hidden)