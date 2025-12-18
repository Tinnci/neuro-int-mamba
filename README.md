# Neuro-INT Mamba

A Bio-inspired Mamba Architecture for Dexterous Manipulation with Intrinsic Neural Timescales (INT).

## Features

- **Dual-Stream INT**: Parallel fast (sensory) and slow (cognitive) streams.
- **Predictive Coding**: Efference copy loop for error-driven learning.
- **Chandelier Gating**: Inhibitory control inspired by Chandelier cells.
- **Spinal Reflex**: Low-level feedback for immediate response.
- **Real-time I/O**: $O(1)$ inference step for closed-loop control.

## Installation

```bash
pip install neuro-int-mamba
```

## Usage

```python
import torch
from neuro_int_mamba import NeuroINTMamba

input_dims = {
    'proprio': 54,
    'tactile': 100,
    'visual': 256,
    'goal': 32
}

model = NeuroINTMamba(input_dims, model_dim=512, num_layers=6)

# Real-time control loop
states = None
predictions = None

while True:
    # Get sensory data
    p, t, v, g = get_sensors() 
    
    # Step model
    motor_cmd, states, predictions = model.step(p, t, v, g, states, predictions)
    
    # Apply motor command
    apply_motor(motor_cmd)
```

## Development

This project uses [Astral](https://astral.sh/) tools for high-performance development:
- **uv**: Package management
- **ruff**: Linting and formatting
- **ty**: Type checking

```bash
uv run ruff check .
uv run ty check .
uv run pytest
```

## Publishing to PyPI

This project is configured to be built with `hatchling` and published via `uv`.

1. **Build the package**:
   ```bash
   uv build
   ```

2. **Publish to PyPI**:
   ```bash
   # Use __token__ as username and your PyPI API token as password
   uv publish
   ```

## License

MIT
