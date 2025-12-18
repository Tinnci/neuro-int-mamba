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
        # x: (B, D)
        batch_size = x.shape[0]
        if state is None:
            state = torch.zeros(batch_size, self.dim, self.d_state, device=x.device)
            
        projected = self.x_proj(x)
        dt, B, C = torch.split(projected, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt) * dt_scale) # (B, D)
        
        A = -torch.exp(self.A_log) # (D, N)
        
        # Simplified SSM discretization
        # dA = exp(dt * A)
        # dB = dt * B
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0)) # (B, D, N)
        dB = dt.unsqueeze(-1) * B.unsqueeze(1) # (B, D, N)
        
        # Update state: h = dA * h + dB * x
        state = dA * state + dB * x.unsqueeze(-1)
        
        # Output: y = C * h + D * x
        y = torch.einsum("bdn,bn->bd", state, C) + x * self.D
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
    Now supports an optional 'intent_prior' (e.g., from EMG) to bias the prediction.
    """
    def __init__(self, dim):
        super().__init__()
        self.block = DualStreamINTBlock(dim)
        self.predictor = nn.Linear(dim, dim)
        self.intent_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        
    def forward(self, x, prev_prediction=None, intent_prior=None):
        # If intent_prior is provided, use it to bias the current state
        if intent_prior is not None:
            gate = self.intent_gate(torch.cat([x, intent_prior], dim=-1))
            x = x * (1 - gate) + intent_prior * gate
            
        if prev_prediction is not None:
            x_error = x - prev_prediction
            h = self.block(x_error)
        else:
            h = self.block(x)
        prediction = self.predictor(h)
        return h, prediction

    def step(self, x, state=None, prev_prediction=None, intent_prior=None):
        if intent_prior is not None:
            gate = self.intent_gate(torch.cat([x, intent_prior], dim=-1))
            x = x * (1 - gate) + intent_prior * gate
            
        if prev_prediction is not None:
            x_error = x - prev_prediction
            h, new_state = self.block.step(x_error, state)
        else:
            h, new_state = self.block.step(x, state)
        prediction = self.predictor(h)
        return h, new_state, prediction

class TactileEncoder(nn.Module):
    """
    Tactile Encoder: Uses 1D Convolution as a spatial prior for array sensors.
    Handles both batch (4D) and step (3D) inputs.
    """
    def __init__(self, input_dim, model_dim):
        super().__init__()
        self.input_dim = input_dim
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(start_dim=-2),
            nn.Linear(16 * input_dim, model_dim)
        )

    def forward(self, x):
        # x shape: (B, L, D) or (B, D)
        if x.dim() == 3:
            B, L, D = x.shape
            x_reshaped = x.view(B * L, 1, D)
            h = self.conv(x_reshaped)
            return h.view(B, L, -1)
        else:
            # x shape: (B, D)
            x_reshaped = x.unsqueeze(1) # (B, 1, D)
            return self.conv(x_reshaped)

class VisualEncoder(nn.Module):
    """
    Visual Encoder: Optimized for high-resolution features.
    Uses a multi-layer perceptron with LayerNorm for stability.
    """
    def __init__(self, input_dim, model_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, model_dim * 2),
            nn.LayerNorm(model_dim * 2),
            nn.GELU(),
            nn.Linear(model_dim * 2, model_dim),
            nn.LayerNorm(model_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class EMGEncoder(nn.Module):
    """
    EMG Encoder: Processes surface Electromyography signals.
    Uses a 1D Conv to capture temporal patterns in muscle activation envelopes.
    """
    def __init__(self, input_dim, model_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Flatten(start_dim=-2),
            nn.Linear(8 * input_dim, model_dim)
        )
        
    def forward(self, x):
        # x: (B, L, D) or (B, D)
        if x.dim() == 3:
            B, L, D = x.shape
            x = x.view(B * L, 1, D)
            return self.conv(x).view(B, L, -1)
        else:
            return self.conv(x.unsqueeze(1))

class SynergyBottleneck(nn.Module):
    """
    Muscle Synergy Bottleneck: Implements dimensionality reduction to extract
    low-dimensional control primitives (synergies) from high-dimensional intent.
    """
    def __init__(self, dim, num_synergies):
        super().__init__()
        self.encoder = nn.Linear(dim, num_synergies)
        self.decoder = nn.Linear(num_synergies, dim)
        
    def forward(self, x):
        synergy = torch.tanh(self.encoder(x))
        return self.decoder(synergy), synergy

class SpinalReflex(nn.Module):
    """
    Spinal Reflex Layer: Simulates low-level, fast feedback loops.
    Implements PD (Proportional-Derivative) control to mimic muscle spindle sensitivity
    to both muscle length (position) and rate of change (velocity).
    """
    def __init__(self, num_dof):
        super().__init__()
        # Kp: Proportional gain (Position)
        self.kp = nn.Parameter(torch.ones(num_dof) * 0.1)
        # Kd: Derivative gain (Velocity)
        self.kd = nn.Parameter(torch.ones(num_dof) * 0.01)
        
    def forward(self, pos, vel, gain_mod=None):
        """
        pos: (B, L, num_dof) or (B, num_dof)
        vel: (B, L, num_dof) or (B, num_dof)
        gain_mod: Optional modulation signal from higher centers (e.g., EMG intent)
        """
        kp = self.kp
        kd = self.kd
        if gain_mod is not None:
            # gain_mod: (B, L, 1) or (B, 1)
            kp = kp * (1.0 + gain_mod)
            kd = kd * (1.0 + gain_mod)
            
        return -(kp * pos + kd * vel)
