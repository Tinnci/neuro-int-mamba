import torch
import torch.nn as nn
from .nn import SpinalReflex, PredictiveCodingLayer

class NeuroINTMamba(nn.Module):
    """
    Full Neuro-INT Mamba Architecture.
    """
    def __init__(self, input_dims, model_dim, num_layers):
        super().__init__()
        # 1. Spinal Reflex: Low-level feedback
        self.spinal_reflex = SpinalReflex(input_dims['proprio'])
        
        # 2. Thalamic Encoder: Multi-modal fusion
        self.proprio_proj = nn.Linear(input_dims['proprio'], model_dim)
        self.tactile_proj = nn.Linear(input_dims['tactile'], model_dim)
        self.visual_proj = nn.Linear(input_dims['visual'], model_dim)
        self.goal_proj = nn.Linear(input_dims['goal'], model_dim)
        
        self.fusion = nn.Linear(model_dim * 4, model_dim)
        
        # 3. Layers with Predictive Coding (Cerebral Cortex simulation)
        self.layers = nn.ModuleList([
            PredictiveCodingLayer(model_dim) for _ in range(num_layers)
        ])
        
        # 4. Output head: Motor commands
        self.motor_head = nn.Linear(model_dim, input_dims['proprio'])
        
    def forward(self, proprio, tactile, visual, goal):
        reflex_cmd = self.spinal_reflex(proprio)
        p = self.proprio_proj(proprio)
        t = self.tactile_proj(tactile)
        v = self.visual_proj(visual)
        g = self.goal_proj(goal)
        
        x = self.fusion(torch.cat([p, t, v, g], dim=-1))
        
        current_input = x
        prediction = None
        for layer in self.layers:
            current_input, prediction = layer(current_input, prediction)
            
        cortical_cmd = self.motor_head(current_input)
        motor_cmd = cortical_cmd + reflex_cmd
        return motor_cmd, prediction

    def step(self, proprio, tactile, visual, goal, states=None, predictions=None):
        if states is None:
            states = [None] * len(self.layers)
        if predictions is None:
            predictions = [None] * len(self.layers)
            
        reflex_cmd = self.spinal_reflex(proprio)
        p = self.proprio_proj(proprio)
        t = self.tactile_proj(tactile)
        v = self.visual_proj(visual)
        g = self.goal_proj(goal)
        
        x = self.fusion(torch.cat([p, t, v, g], dim=-1))
        
        current_input = x
        new_states = []
        new_predictions = []
        for i, layer in enumerate(self.layers):
            h, s, pred = layer.step(current_input, states[i], predictions[i])
            current_input = h
            new_states.append(s)
            new_predictions.append(pred)
            
        cortical_cmd = self.motor_head(current_input)
        motor_cmd = cortical_cmd + reflex_cmd
        return motor_cmd, new_states, new_predictions
