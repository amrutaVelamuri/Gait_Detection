import torch
import torch.nn as nn

class DualStreamModel(nn.Module):
    def __init__(self, hand_feat_dim, input_length=100):
        super().__init__()
        self.grf_cnn = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2), nn.Flatten())
        self.cop_cnn = nn.Sequential(
            nn.Conv1d(2, 16, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2), nn.Flatten())
        self.fc_hand = nn.Sequential(
            nn.Linear(hand_feat_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.3))

        with torch.no_grad():
            dummy_grf = torch.zeros(1, 1, input_length)
            dummy_cop = torch.zeros(1, 2, input_length)
            grf_feat_size = self.grf_cnn(dummy_grf).shape[1]
            cop_feat_size = self.cop_cnn(dummy_cop).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(grf_feat_size + cop_feat_size + 64, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2))

    def forward(self, grf, cop, hand):
        grf_feat = self.grf_cnn(grf)
        cop_feat = self.cop_cnn(cop)
        hand_feat = self.fc_hand(hand)
        combined = torch.cat([grf_feat, cop_feat, hand_feat], dim=1)
        return self.classifier(combined)
