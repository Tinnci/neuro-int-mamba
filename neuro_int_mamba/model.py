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
        self.proprio_proj = nn.Linear(self.num_dof * 2, d_model)
        self.tactile_conv = TactileEncoder(tactile_dim, d_model)
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
        self.motor_head = nn.Linear(d_model, self.num_dof)
        
        # State management for step()
        self.states: Optional[List[Any]] = None
        self.predictions: Optional[List[Optional[Tensor]]] = None

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
        Real-time O(1) inference step.
        """
        if self.states is None:
            self.reset_states()
            
        # If proprio is not provided, assume zero (or use last state if available)
        if proprio is None:
            proprio = torch.zeros(1, self.num_dof * 2, device=visual.device)
            
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
        model_dim = self.proprio_proj.out_features
        self.subject_adapters[subject_id] = SubjectAdapter(model_dim, bottleneck_dim)

    def reset_states(self):
        """
        Resets the internal states for real-time inference.
        """
        self.states: Optional[List[Any]] = [None] * len(self.layers)
        self.predictions: Optional[List[Optional[Tensor]]] = [cast(Optional[Tensor], None)] * len(self.layers)
