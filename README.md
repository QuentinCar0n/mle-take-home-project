# Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Setup Modal:

```bash
modal setup
```

# Usage

1. Get your huggingface token: https://huggingface.co/settings/tokens

2. Add your huggingface token at the top of inference.py file:

```python
HF_TOKEN = "<your-huggingface-token>"
```

3. Run inference:

```bash
modal run inference.py
```
