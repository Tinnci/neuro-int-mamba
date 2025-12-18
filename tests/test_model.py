import torch
from neuro_int_mamba import NeuroINTMamba

def test_model_forward():
    model = NeuroINTMamba(
        vision_dim=128, 
        tactile_dim=16, 
        emg_dim=8, 
        action_dim=2, 
        d_model=128, 
        num_layers=2
    )
    
    p = torch.randn(1, 5, 4)
    t = torch.randn(1, 5, 16)
    v = torch.randn(1, 5, 3, 128, 128)
    
    motor_cmd, next_pred = model(p, t, v)
    assert motor_cmd.shape == (1, 5, 2)

def test_model_step():
    model = NeuroINTMamba(
        vision_dim=128, 
        tactile_dim=16, 
        emg_dim=8, 
        action_dim=2, 
        d_model=128, 
        num_layers=2
    )
    
    p_t = torch.randn(1, 4)
    t_t = torch.randn(1, 16)
    v_t = torch.randn(1, 3, 128, 128)
    
    model.reset_states()
    motor_out, prediction = model.step(visual=v_t, tactile=t_t, proprio=p_t)
    assert motor_out.shape == (1, 2)
    assert prediction.shape == (1, 128)

def test_model_reset():
    model = NeuroINTMamba(num_layers=2)
    model.reset_states()
    assert model.states is not None
    assert all(s is None for s in model.states)
    assert model.predictions is not None
    assert all(p is None for p in model.predictions)

def test_model_emg():
    model = NeuroINTMamba(
        vision_dim=128, 
        tactile_dim=16, 
        emg_dim=8, 
        action_dim=2, 
        d_model=128, 
        num_layers=2, 
        use_emg=True
    )
    
    p = torch.randn(1, 5, 4)
    t = torch.randn(1, 5, 16)
    v = torch.randn(1, 5, 3, 128, 128)
    e = torch.randn(1, 5, 8)
    
    motor_cmd, _ = model(p, t, v, emg=e)
    assert motor_cmd.shape == (1, 5, 2)
    
    # Test step with EMG
    p_t = torch.randn(1, 4)
    t_t = torch.randn(1, 16)
    v_t = torch.randn(1, 3, 128, 128)
    e_t = torch.randn(1, 8)
    
    model.reset_states()
    motor_out, _ = model.step(visual=v_t, tactile=t_t, emg=e_t, proprio=p_t)
    assert motor_out.shape == (1, 2)
