import torch
import torch.nn as nn
from .nn import SpinalReflex, PredictiveCodingLayer, TactileEncoder, VisualEncoder, EMGEncoder, SynergyBottleneck, SubjectAdapter

class NeuroINTMamba(nn.Module):
    """
    Full Neuro-INT Mamba Architecture.
    """
    def __init__(self, input_dims, model_dim, num_layers, use_emg=False):
        super().__init__()
        self.use_emg = use_emg
        # 1. Spinal Reflex: Low-level PD feedback
        # Assuming proprio input is [pos, vel], so num_dof is half of proprio_dim
        self.num_dof = input_dims['proprio'] // 2
        self.spinal_reflex = SpinalReflex(self.num_dof)
        
        # 2. Thalamic Encoder: Multi-modal fusion
        self.proprio_proj = nn.Linear(input_dims['proprio'], model_dim)
        
        # Tactile Encoder: Using a 1D Conv as a spatial prior for array sensors
        self.tactile_conv = TactileEncoder(input_dims['tactile'], model_dim)
        
        self.visual_proj = VisualEncoder(input_dims['visual'], model_dim)
        self.goal_proj = nn.Linear(input_dims['goal'], model_dim)
        
        # Optional EMG Encoder for Transfer Learning
        if self.use_emg:
            self.emg_encoder = EMGEncoder(input_dims['emg'], model_dim)
            self.synergy_bottleneck = SynergyBottleneck(model_dim, num_synergies=8)
            # Subject Adapters for multi-user fine-tuning
            self.subject_adapters = nn.ModuleDict()
            self.fusion = nn.Linear(model_dim * 5, model_dim)
        else:
            self.fusion = nn.Linear(model_dim * 4, model_dim)
        
        # 3. Layers with Predictive Coding (Cerebral Cortex simulation)
        self.layers = nn.ModuleList([
            PredictiveCodingLayer(model_dim) for _ in range(num_layers)
        ])
        
        # 4. Output head: Motor commands (Target positions/torques for DOFs)
        self.motor_head = nn.Linear(model_dim, self.num_dof)
        
    def forward(self, proprio, tactile, visual, goal, emg=None, subject_id=None):
        # Split proprioception into position and velocity
        pos, vel = torch.split(proprio, self.num_dof, dim=-1)
        
        # 1. Low-level Reflex (PD Control)
        # If EMG is provided, use its intensity to modulate reflex gains
        gain_mod = None
        if self.use_emg and emg is not None:
            e_feat = self.emg_encoder(emg)
            
            # Apply subject-specific adapter if provided
            if subject_id is not None and subject_id in self.subject_adapters:
                e_feat = self.subject_adapters[subject_id](e_feat)
                
            _, synergy = self.synergy_bottleneck(e_feat)
            gain_mod = torch.mean(synergy, dim=-1, keepdim=True)
            
        reflex_cmd = self.spinal_reflex(pos, vel, gain_mod=gain_mod)
        
        # 2. Thalamic Encoding & Fusion
        p = self.proprio_proj(proprio)
        t = self.tactile_conv(tactile)
        v = self.visual_proj(visual)
        g = self.goal_proj(goal)
        
        if self.use_emg and emg is not None:
            e = self.emg_encoder(emg)
            if subject_id is not None and subject_id in self.subject_adapters:
                e = self.subject_adapters[subject_id](e)
            x = self.fusion(torch.cat([p, t, v, g, e], dim=-1))
        else:
            x = self.fusion(torch.cat([p, t, v, g], dim=-1))
        
        # 3. Sequential Processing with Feedback (Predictive Coding)
        current_input = x
        prediction = None
        
        # Use EMG as an intent prior for the first layer if available
        intent_prior = None
        if self.use_emg and emg is not None:
            intent_prior = self.emg_encoder(emg)
            if subject_id is not None and subject_id in self.subject_adapters:
                intent_prior = self.subject_adapters[subject_id](intent_prior)
            
        for i, layer in enumerate(self.layers):
            # Only the first layer or all layers can receive the intent prior
            # Here we pass it to all layers to maintain the 'intent' throughout the hierarchy
            current_input, prediction = layer(current_input, prediction, intent_prior=intent_prior)
            
        # 4. Motor Output (Cortical command + Spinal reflex)
        cortical_cmd = self.motor_head(current_input)
        motor_cmd = cortical_cmd + reflex_cmd
        return motor_cmd, prediction

    def step(self, proprio, tactile, visual, goal, emg=None, subject_id=None, states=None, predictions=None):
        if states is None:
            states = [None] * len(self.layers)
        if predictions is None:
            predictions = [None] * len(self.layers)
            
        # Split proprioception
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
        g = self.goal_proj(goal)
        
        if self.use_emg and emg is not None:
            e = self.emg_encoder(emg)
            if subject_id is not None and subject_id in self.subject_adapters:
                e = self.subject_adapters[subject_id](e)
            x = self.fusion(torch.cat([p, t, v, g, e], dim=-1))
        else:
            x = self.fusion(torch.cat([p, t, v, g], dim=-1))
        
        # 3. Cortical Processing (Predictive Coding)
        current_input = x
        new_states = []
        new_predictions = []
        
        intent_prior = None
        if self.use_emg and emg is not None:
            intent_prior = self.emg_encoder(emg)
            if subject_id is not None and subject_id in self.subject_adapters:
                intent_prior = self.subject_adapters[subject_id](intent_prior)
            
        for i, layer in enumerate(self.layers):
            h, s, pred = layer.step(current_input, states[i], predictions[i], intent_prior=intent_prior)
            current_input = h
            new_states.append(s)
            new_predictions.append(pred)
            
        # 4. Motor Output
        cortical_cmd = self.motor_head(current_input)
        motor_cmd = cortical_cmd + reflex_cmd
        return motor_cmd, new_states, new_predictions

    def add_subject_adapter(self, subject_id, bottleneck_dim=64):
        """
        Adds a new subject-specific adapter to the model.
        """
        model_dim = self.proprio_proj.out_features
        self.subject_adapters[subject_id] = SubjectAdapter(model_dim, bottleneck_dim)

    def reset_states(self, batch_size=1, device=None):
        """
        Returns initial states and predictions for the model.
        If device is None, it uses the device of the model parameters.
        """
        if device is None:
            device = next(self.parameters()).device
            
        states = [None] * len(self.layers)
        predictions = [None] * len(self.layers)
        return states, predictions
