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
    assert motor_cmd.shape == (1, 5, 27)

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
    assert motor_out.shape == (1, 27)
    assert len(states) == 2
    assert len(predictions) == 2

def test_model_reset():
    input_dims = {'proprio': 54, 'tactile': 100, 'visual': 256, 'goal': 32}
    model = NeuroINTMamba(input_dims, model_dim=128, num_layers=2)
    states, preds = model.reset_states()
    assert all(s is None for s in states)
    assert all(p is None for p in preds)

def test_model_emg():
    input_dims = {
        'proprio': 54,
        'tactile': 100,
        'visual': 256,
        'goal': 32,
        'emg': 8
    }
    model = NeuroINTMamba(input_dims, model_dim=128, num_layers=2, use_emg=True)
    
    p = torch.randn(1, 5, 54)
    t = torch.randn(1, 5, 100)
    v = torch.randn(1, 5, 256)
    g = torch.randn(1, 5, 32)
    e = torch.randn(1, 5, 8)
    
    motor_cmd, _ = model(p, t, v, g, emg=e)
    assert motor_cmd.shape == (1, 5, 27)
    
    # Test step with EMG
    p_t = torch.randn(1, 54)
    t_t = torch.randn(1, 100)
    v_t = torch.randn(1, 256)
    g_t = torch.randn(1, 32)
    e_t = torch.randn(1, 8)
    
    motor_out, _, _ = model.step(p_t, t_t, v_t, g_t, emg=e_t)
    assert motor_out.shape == (1, 27)
