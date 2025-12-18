import torch
from neuro_int_mamba import NeuroINTMamba

def test_model_forward():
    input_dims = {
        'proprio': 54,
        'tactile': 100,
        'visual': 256,
        'goal': 32
    }
    model = NeuroINTMamba(input_dims, model_dim=128, num_layers=2)
    
    p = torch.randn(1, 5, 54)
    t = torch.randn(1, 5, 100)
    v = torch.randn(1, 5, 256)
    g = torch.randn(1, 5, 32)
    
    motor_cmd, next_pred = model(p, t, v, g)
    assert motor_cmd.shape == (1, 5, 54)

def test_model_step():
    input_dims = {
        'proprio': 54,
        'tactile': 100,
        'visual': 256,
        'goal': 32
    }
    model = NeuroINTMamba(input_dims, model_dim=128, num_layers=2)
    
    p_t = torch.randn(1, 54)
    t_t = torch.randn(1, 100)
    v_t = torch.randn(1, 256)
    g_t = torch.randn(1, 32)
    
    motor_out, states, predictions = model.step(p_t, t_t, v_t, g_t)
    assert motor_out.shape == (1, 54)
    assert len(states) == 2
    assert len(predictions) == 2
