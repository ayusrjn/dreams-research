# Installation

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/ayush-ranjan/dreams-research.git
cd dreams-research
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# or
.\venv\Scripts\activate   # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Additional Dependencies

For caption embeddings (Phase 2B):

```bash
pip install sentence-transformers>=2.2.0
```

## Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import clip; print('CLIP: OK')"
```
