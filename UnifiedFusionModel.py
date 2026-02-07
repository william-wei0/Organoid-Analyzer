import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Attention(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, lstm_out):
        scores = self.attn(lstm_out).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        
        context = torch.sum(weights.unsqueeze(-1) * lstm_out, dim=1)
        context = self.dropout(context)
        return context, weights

class UnifiedFusionModel(nn.Module):
    def __init__(self, seq_input_size, track_input_size, hidden_size=16, fusion_size=32, dropout=0.5):
        super().__init__()
        self.track_input_size = track_input_size

        self.lstm = nn.LSTM(input_size=seq_input_size, hidden_size=hidden_size,
                            batch_first=True, bidirectional=False)
        self.attn = Attention(hidden_size, dropout)
        self.norm = nn.LayerNorm(hidden_size)

        track_output_size = hidden_size

        print("Total Sequence parameters:", count_params(self.lstm) + count_params(self.attn) + count_params(self.norm))

        if track_input_size > 0:
            self.track_fc = nn.Sequential(
                nn.Linear(track_input_size,16),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(16, track_output_size),
            )
            print("Total Track parameters:", count_params(self.track_fc))

            self.use_track = True
        else:
            self.use_track = False

        self.fusion_fc = nn.Sequential(
            nn.Linear(hidden_size + track_output_size, fusion_size),
            nn.ReLU(),
            nn.Linear(fusion_size, 3)
        )

    def forward(self, x_seq, x_track, lstm_weight=0.5):
        lstm_out, _ = self.lstm(x_seq)
        lstm_out = self.norm(lstm_out)
        lstm_feat, attn_weights = self.attn(lstm_out)
        lstm_feat = lstm_feat * lstm_weight
        

        if self.use_track:
            track_feat = self.track_fc(x_track)
            track_feat = track_feat * (1 - lstm_weight)
            fused = torch.cat([lstm_feat, track_feat], dim=1)
        else:
            fused = lstm_feat
        return self.fusion_fc(fused)
    
    def run_inference(self, X_seq, X_track, device):
        with torch.no_grad():
            logits = self.forward(X_seq.to(device), X_track.to(device))
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
        return preds, probs
    
    def extract_LSTM_track_features(self, x_seq, x_track, lstm_weight=0.5):
        lstm_out, _ = self.lstm(x_seq)
        lstm_out = self.norm(lstm_out)
        lstm_feat, attn_weights = self.attn(lstm_out)
        lstm_feat = lstm_feat * lstm_weight

        if self.use_track:
            track_feat = self.track_fc(x_track)
            track_feat = track_feat * (1 - lstm_weight)
            fused = torch.cat([lstm_feat, track_feat], dim=1)
        else:
            fused = lstm_feat

        return fused.detach().cpu()
        
def count_params(module):
    return sum(p.numel() for p in module.parameters())