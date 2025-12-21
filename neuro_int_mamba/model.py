from typing import Optional, List, TYPE_CHECKING, cast, Any
import torch
from torch import Tensor
import torch.nn as nn
from .nn import SpinalReflex, PredictiveCodingLayer, TactileEncoder, VisualEncoder, EMGEncoder, SynergyBottleneck, SubjectAdapter

if TYPE_CHECKING:
    from typing import Any

class NeuroINTMamba(nn.Module):
    """
    Full Neuro-INT Mamba Architecture.
    """
    if TYPE_CHECKING:
        use_emg: bool
        num_dof: int
        d_model: int
        num_layers: int
        states: Optional[List[Any]]
        predictions: Optional[List[Optional[Tensor]]]

    def __init__(
        self,
        vision_dim: int = 128,
        tactile_dim: int = 16,
        emg_dim: int = 8,
        action_dim: int = 2,
        d_model: int = 128,
        num_layers: int = 4,
        use_emg: bool = False,
    ):
        super().__init__()
        self.use_emg: bool = use_emg
        self.num_dof: int = action_dim
        self.d_model: int = d_model
        self.num_layers: int = num_layers
        
        # 1. Spinal Reflex: Low-level PD feedback
        # Proprioception is assumed to be [pos, vel] for each DOF
        self.spinal_reflex = SpinalReflex(self.num_dof)
        
        # 2. Thalamic Encoder: Multi-modal fusion
        self.proprio_proj = nn.Sequential(
            nn.Linear(self.num_dof * 2, d_model),
            nn.LayerNorm(d_model)
        )
        self.tactile_conv = nn.Sequential(
            TactileEncoder(tactile_dim, d_model),
            nn.LayerNorm(d_model)
        )
        self.visual_proj = VisualEncoder(d_model)
        
        # Optional EMG Encoder for Transfer Learning
        if self.use_emg:
            self.emg_encoder = EMGEncoder(emg_dim, d_model)
            self.synergy_bottleneck = SynergyBottleneck(d_model, num_synergies=8)
            self.subject_adapters = nn.ModuleDict()
            self.fusion = nn.Linear(d_model * 4, d_model)
        else:
            self.fusion = nn.Linear(d_model * 3, d_model)
        
        # 3. Layers with Predictive Coding (Cerebral Cortex simulation)
        self.layers = nn.ModuleList([
            PredictiveCodingLayer(d_model) for _ in range(num_layers)
        ])
        
        # 4. Output head: Motor commands
        self.motor_head = nn.Sequential(
            nn.Linear(d_model, self.num_dof),
            nn.Tanh() # Strictly bound cortical commands to [-1, 1]
        )
        
        # State management for step()
        self.states: Optional[List[Any]] = None
        self.predictions: Optional[List[Optional[Tensor]]] = None
        
        # Initialize all weights to be small for stability
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, proprio, tactile, visual, emg=None, subject_id=None):
        # Split proprioception into position and velocity
        pos, vel = torch.split(proprio, self.num_dof, dim=-1)
        
        # 1. Low-level Reflex (PD Control)
        gain_mod = None
        if self.use_emg and emg is not None:
            e_feat = self.emg_encoder(emg)
            if subject_id is not None and subject_id in self.subject_adapters:
                e_feat = self.subject_adapters[subject_id](e_feat)
            _, synergy = self.synergy_bottleneck(e_feat)
            gain_mod = torch.mean(synergy, dim=-1, keepdim=True)
            
        reflex_cmd = self.spinal_reflex(pos, vel, gain_mod=gain_mod)
        
        # 2. Thalamic Encoding & Fusion
        p = self.proprio_proj(proprio)
        t = self.tactile_conv(tactile)
        v = self.visual_proj(visual)
        
        if self.use_emg and emg is not None:
            e = self.emg_encoder(emg)
            if subject_id is not None and subject_id in self.subject_adapters:
                e = self.subject_adapters[subject_id](e)
            x = self.fusion(torch.cat([p, t, v, e], dim=-1))
        else:
            x = self.fusion(torch.cat([p, t, v], dim=-1))
        
        # 3. Sequential Processing with Feedback (Predictive Coding)
        current_input = x
        prediction = None
        
        intent_prior = None
        if self.use_emg and emg is not None:
            intent_prior = self.emg_encoder(emg)
            if subject_id is not None and subject_id in self.subject_adapters:
                intent_prior = self.subject_adapters[subject_id](intent_prior)
            
        for layer in self.layers:
            if TYPE_CHECKING:
                assert isinstance(layer, PredictiveCodingLayer)
            current_input, prediction = layer(current_input, prediction, intent_prior=intent_prior)
            
        # 4. Motor Output
        cortical_cmd = self.motor_head(current_input)
        motor_cmd = cortical_cmd + reflex_cmd
        return motor_cmd, prediction

    def step(self, visual, tactile, emg=None, proprio=None, subject_id=None):
        """
        Real-time O(1) inference step. Supports batching.
        """
        if self.states is None:
            self.reset_states()
            
        batch_size = visual.shape[0]
        device = visual.device

        # If proprio is not provided, assume zero
        if proprio is None:
            proprio = torch.zeros(batch_size, self.num_dof * 2, device=device)
            
        pos, vel = torch.split(proprio, self.num_dof, dim=-1)
        
        # 1. Low-level Reflex
        gain_mod = None
        if self.use_emg and emg is not None:
            e_feat = self.emg_encoder(emg)
            if subject_id is not None and subject_id in self.subject_adapters:
                e_feat = self.subject_adapters[subject_id](e_feat)
            _, synergy = self.synergy_bottleneck(e_feat)
            gain_mod = torch.mean(synergy, dim=-1, keepdim=True)
            
        reflex_cmd = self.spinal_reflex(pos, vel, gain_mod=gain_mod)
        
        # 2. Thalamic Encoding
        p = self.proprio_proj(proprio)
        t = self.tactile_conv(tactile)
        v = self.visual_proj(visual)
        
        if self.use_emg and emg is not None:
            e = self.emg_encoder(emg)
            if subject_id is not None and subject_id in self.subject_adapters:
                e = self.subject_adapters[subject_id](e)
            x = self.fusion(torch.cat([p, t, v, e], dim=-1))
        else:
            x = self.fusion(torch.cat([p, t, v], dim=-1))
        
        # 3. Cortical Processing (Predictive Coding)
        current_input = x
        new_states: List[Any] = []
        new_predictions: List[Optional[Tensor]] = []
        
        intent_prior = None
        if self.use_emg and emg is not None:
            intent_prior = self.emg_encoder(emg)
            if subject_id is not None and subject_id in self.subject_adapters:
                intent_prior = self.subject_adapters[subject_id](intent_prior)
            
        # Ensure states are initialized
        states = self.states if self.states is not None else [None] * len(self.layers)
        preds = self.predictions if self.predictions is not None else [cast(Optional[Tensor], None)] * len(self.layers)

        for i, layer in enumerate(self.layers):
            if TYPE_CHECKING:
                assert isinstance(layer, PredictiveCodingLayer)
            h, s, pred = layer.step(current_input, states[i], preds[i], intent_prior=intent_prior)
            current_input = h
            new_states.append(s)
            new_predictions.append(pred)
            
        self.states: Optional[List[Any]] = new_states
        self.predictions: Optional[List[Optional[Tensor]]] = new_predictions
        
        # 4. Motor Output
        cortical_cmd = self.motor_head(current_input)
        motor_cmd = cortical_cmd + reflex_cmd
        return motor_cmd, new_predictions[-1]

    def add_subject_adapter(self, subject_id, bottleneck_dim=64):
        """
        Adds a new subject-specific adapter to the model.
        """
        # Get d_model from the fusion layer or similar
        model_dim = self.d_model
        self.subject_adapters[subject_id] = SubjectAdapter(model_dim, bottleneck_dim)

    def reset_states(self, batch_size=1, device=None):
        """
        Resets the internal states for real-time inference.
        """
        # We keep them as None, the layers will initialize them on the first step
        # based on the input batch size.
        self.states = [None] * len(self.layers)
        self.predictions = [None] * len(self.layers)
