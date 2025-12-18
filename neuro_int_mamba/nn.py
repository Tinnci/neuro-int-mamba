import torch
import torch.nn as nn
import torch.nn.functional as F

class ChandelierGating(nn.Module):
    """
    Chandelier Gating Mechanism: Mimics inhibitory interneurons (Chandelier cells)
    that regulate pyramidal neuron firing based on activity levels.
    """
    def __init__(self, dim, alpha=1.0, beta=0.1):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))
        
    def forward(self, x):
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        gate = torch.sigmoid(self.alpha - self.beta * (norm ** 2))
        return x * gate

class ThalamicMixer(nn.Module):
    """
    Thalamic Mixer: Dynamically fuses fast and slow streams based on task context.
    """
    def __init__(self, dim):
        super().__init__()
        self.mixer = nn.Linear(dim * 2, dim)
        self.gate = nn.Linear(dim * 2, 2)
        
    def forward(self, h_fast, h_slow):
        combined = torch.cat([h_fast, h_slow], dim=-1)
        weights = torch.softmax(self.gate(combined), dim=-1)
        w_fast = weights[..., 0:1]
        w_slow = weights[..., 1:2]
        return w_fast * h_fast + w_slow * h_slow

class SimpleMambaBlock(nn.Module):
    """
    A simplified Mamba-like block for demonstration.
    """
    def __init__(self, dim, dt_rank, d_state):
        super().__init__()
        self.dim = dim
        self.dt_rank = dt_rank
        self.d_state = d_state
        self.x_proj = nn.Linear(dim, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, dim, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float().repeat(dim, 1)))
        self.D = nn.Parameter(torch.ones(dim))
        
    def forward(self, x, dt_scale=1.0):
        batch, seq_len, dim = x.shape
        projected = self.x_proj(x)
        dt, B, C = torch.split(projected, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt) * dt_scale
        dt = F.softplus(dt)
        A = -torch.exp(self.A_log)
        y = x * torch.exp(A.mean(dim=-1) * dt) + x * self.D
        return y

    def step(self, x, state=None, dt_scale=1.0):
        projected = self.x_proj(x)
        dt, B, C = torch.split(projected, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt) * dt_scale
        dt = F.softplus(dt)
        A = -torch.exp(self.A_log)
        y = x * torch.exp(A.mean(dim=-1) * dt) + x * self.D
        return y, state

class DualStreamINTBlock(nn.Module):
    """
    Dual-Stream INT Block: Parallel fast and slow streams with different time constants.
    """
    def __init__(self, dim, dt_rank=16, d_state=16):
        super().__init__()
        self.fast_mamba = SimpleMambaBlock(dim, dt_rank, d_state)
        self.slow_mamba = SimpleMambaBlock(dim, dt_rank, d_state)
        self.mixer = ThalamicMixer(dim)
        self.gating = ChandelierGating(dim)
        
    def forward(self, x):
        h_fast = self.fast_mamba(x, dt_scale=0.1)
        h_slow = self.slow_mamba(x, dt_scale=2.0)
        h_mixed = self.mixer(h_fast, h_slow)
        return self.gating(h_mixed)

    def step(self, x, states=None):
        s_fast, s_slow = states if states is not None else (None, None)
        h_fast, s_fast = self.fast_mamba.step(x, s_fast, dt_scale=0.1)
        h_slow, s_slow = self.slow_mamba.step(x, s_slow, dt_scale=2.0)
        h_mixed = self.mixer(h_fast, h_slow)
        h_gated = self.gating(h_mixed)
        return h_gated, (s_fast, s_slow)

class PredictiveCodingLayer(nn.Module):
    """
    Predictive Coding Layer: Implements the Efference Copy Loop.
    """
    def __init__(self, dim):
        super().__init__()
        self.block = DualStreamINTBlock(dim)
        self.predictor = nn.Linear(dim, dim)
        
    def forward(self, x, prev_prediction=None):
        if prev_prediction is not None:
            x_error = x - prev_prediction
            h = self.block(x_error)
        else:
            h = self.block(x)
        prediction = self.predictor(h)
        return h, prediction

    def step(self, x, state=None, prev_prediction=None):
        if prev_prediction is not None:
            x_error = x - prev_prediction
            h, new_state = self.block.step(x_error, state)
        else:
            h, new_state = self.block.step(x, state)
        prediction = self.predictor(h)
        return h, new_state, prediction

class SpinalReflex(nn.Module):
    """
    Spinal Reflex Layer: Simulates low-level, fast feedback loops.
    """
    def __init__(self, proprio_dim):
        super().__init__()
        self.reflex_gain = nn.Parameter(torch.ones(proprio_dim) * 0.1)
        
    def forward(self, proprio):
        return -self.reflex_gain * proprio
