import torch
import torch.nn as nn


class DictionaryExpert(nn.Module):
    def __init__(self, input_dim, code_dim):
        super().__init__()
        self.dictionary = nn.Parameter(torch.randn(input_dim, code_dim) * 0.01)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2 * code_dim),
            nn.ReLU(),
            nn.Linear(2 * code_dim, code_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = z @ self.dictionary.T
        return recon, z


class GatingNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, K):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, K)
        )

    def forward(self, x):
        logits = self.net(x)
        probs = torch.softmax(logits, dim=1)
        return probs