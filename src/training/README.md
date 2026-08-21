# 🧠 Artifact Training Module

Build GPT from ground up with live telemetry, smart data ingestion, and comprehensive monitoring.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install rich pandas aiohttp
```

### 2. Train a Model
```bash
# Train with data sources
artifact train https://api.example.com/data https://huggingface.co/dataset.json

# Or with local files
artifact train ./data/file1.json ./data/file2.csv
```

### 3. Build from Scratch
```bash
# Create model architecture
artifact build --name my-model --size 7B

# Train the model
artifact train ./data --name my-model --size 7B
```

### 4. Deploy Model
```bash
# Serve model as API
artifact serve ./artifacts/my-model-final.pth --port 8000
```

## 📚 Features

### 🎯 Build GPT from Ground Up
- Start with scratch architecture
- Multi-layer transformer design
- Configurable model sizes (1B to 70B parameters)
- Automatic layer initialization

### 🌊 Smart Data Streaming
- Stream from APIs, URLs, databases, files
- Automatic data type detection
- Intelligent normalization
- Quality scoring and validation

### 📊 Live Telemetry
- Real-time metrics dashboard
- Loss and accuracy tracking
- GPU utilization monitoring
- Token generation speed
- Estimated time remaining

### 🔧 Comprehensive Training
- Mixed precision training (FP16/BF16)
- Gradient accumulation
- Learning rate scheduling
- Checkpoint management
- Distributed training support

## 💻 Terminal UI

The training module features a beautiful, comprehensive terminal UI with:

### Live Dashboard
```
🚀 Training Phase: 🔥 Training
┌─────────────────────────────────────┐
│ 📊 Training Metrics                  │
├─────────────────────────────────────┤
│ Loss           0.8452               │
│ Accuracy       78.45%               │
│ Learning Rate  3.2e-04              │
│ Tokens/sec   2,847                │
│ GPU Util       75.3%                │
│ Memory         15.2GB              │
│ Elapsed        127s                 │
│ ETA            43s                  │
└─────────────────────────────────────┘

[████████████████░░░░░░░░░░] 72.5%
```

### Data Ingestion
```
📥 Data Ingestion
Source: https://api.example.com/data
Type: JSON
Samples: 10,000
Quality: 95.2%
Status: ✅ Success
```

### Model Building
```
🏗️ Model Architecture
Parameters: 7.0B
Layers: 32
Hidden Size: 4096
Attention Heads: 32
Vocab Size: 50257
```

## 📖 Usage Examples

### Basic Training
```python
from artifact.training import ArtifactTrainingEngine, TrainingConfig

config = TrainingConfig(
    model_name="my-gpt",
    model_size="7B",
    batch_size=32,
    learning_rate=3e-4,
    epochs=3
)

engine = ArtifactTrainingEngine(config)
results = await engine.full_pipeline([
    "https://huggingface.co/dataset.json",
    "./local_data.csv"
])
```

### Advanced Configuration
```python
config = TrainingConfig(
    model_name="large-gpt",
    model_size="13B",
    dataset_path="./data",
    output_dir="./artifacts",
    batch_size=64,
    learning_rate=2e-4,
    epochs=5,
    warmup_steps=500,
    max_seq_length=4096,
    gradient_accumulation_steps=8,
    mixed_precision="bf16",
    use_8bit=True,
    lora_r=16,
    lora_alpha=32
)
```

### Data Streaming
```python
from artifact.training import DataStreamer

async with DataStreamer() as streamer:
    # Stream from API
    api_data = await streamer.stream_from_api(
        "https://api.example.com/data",
        headers={"Authorization": "Bearer token"}
    )
    
    # Stream from URL
    web_data = await streamer.stream_from_url("https://example.com/data.json")
    
    # Stream from file
    file_data = await streamer.stream_from_file("./data.json")
```

## 🛠️ Configuration

### TrainingConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model_name | str | "artifact-gpt" | Model identifier |
| model_size | str | "7B" | Model size (1B, 2B, 3B, 6B, 7B, 13B, 30B, 70B) |
| dataset_path | str | "" | Path to training data |
| output_dir | str | "./artifacts" | Output directory for checkpoints |
| batch_size | int | 32 | Training batch size |
| learning_rate | float | 3e-4 | Learning rate |
| epochs | int | 3 | Number of training epochs |
| warmup_steps | int | 100 | Warmup steps for LR scheduler |
| max_seq_length | int | 2048 | Maximum sequence length |
| gradient_accumulation_steps | int | 4 | Gradient accumulation |
| mixed_precision | str | "fp16" | Mixed precision (fp16, bf16, fp32) |
| optimizer | str | "adamw" | Optimizer (adamw, sgd, lion) |
| scheduler | str | "cosine" | LR scheduler (cosine, linear, constant) |
| seed | int | 42 | Random seed |
| checkpoint_steps | int | 500 | Checkpoint every N steps |
| save_total_limit | int | 3 | Maximum checkpoints to keep |
| logging_steps | int | 10 | Log every N steps |
| eval_steps | int | 250 | Evaluate every N steps |
| test_split | float | 0.1 | Test set split ratio |
| use_8bit | bool | False | Use 8-bit quantization |
| use_deepspeed | bool | False | Use DeepSpeed for distributed training |
| lora_r | int | 8 | LoRA rank |
| lora_alpha | int | 16 | LoRA alpha |
| lora_dropout | float | 0.1 | LoRA dropout |

## 🔧 Architecture

### Model Building from Scratch
```
Input Tokens → Embedding Layer → Transformer Blocks → Output Layer
     ↓              ↓                   ↓                ↓
  [B, S]      [B, S, H]           [B, S, H]        [B, S, V]
  
B = Batch Size
S = Sequence Length
H = Hidden Size
V = Vocabulary Size
```

### Transformer Block
```
Input → Multi-Head Attention → Add & Norm → MLP → Add & Norm → Output
        ↓                     ↓            ↓         ↓
     [B, S, H]             [B, S, H]    [B, S, H]  [B, S, H]
```

## 📈 Monitoring

### Live Metrics
- **Loss**: Training loss (decreasing)
- **Accuracy**: Validation accuracy (increasing)
- **Learning Rate**: Current LR value
- **Tokens/sec**: Generation speed
- **GPU Util**: GPU memory utilization
- **Memory**: VRAM usage
- **Elapsed**: Training time
- **ETA**: Estimated time remaining

### Checkpoint Management
```
./artifacts/
├── my-model-init.pth      # Initial checkpoint
├── my-model-step-500.pth  # Step 500
├── my-model-step-1000.pth # Step 1000
└── my-model-final.pth     # Final model
```

## 🌐 Data Sources

### Supported Sources
- **API Endpoints**: `https://api.example.com/data`
- **HuggingFace**: `https://huggingface.co/dataset.json`
- **CSV Files**: `./data.csv`
- **JSON Files**: `./data.json`
- **Parquet Files**: `./data.parquet`
- **SQL Queries**: `sql:SELECT * FROM table`
- **Database**: `db:postgresql://...`

### Smart Detection
The system automatically detects data types:
- JSON objects and arrays
- CSV tabular data
- XML/HTML documents
- Plain text
- Binary formats

### Normalization
All data is normalized to a standard format:
```python
[
    {"text": "Sample text 1"},
    {"text": "Sample text 2"},
    {"metadata": {"key": "value"}, "text": "Sample text 3"}
]
```

## 🚀 Deployment

### Serve as API
```bash
artifact serve ./artifacts/my-model-final.pth --port 8000
```

### API Endpoints
```
POST /v1/completions
POST /v1/chat/completions
GET /v1/models
GET /v1/health
```

### Example API Call
```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-model",
    "prompt": "Hello, world!",
    "max_tokens": 100
  }'
```

## 📚 Documentation

- **CLI Help**: `artifact --help`
- **Train Help**: `artifact train --help`
- **Build Help**: `artifact build --help`
- **Serve Help**: `artifact serve --help`

## 🤝 Contributing

This module is part of the Artifact Engine. For issues and feature requests, please visit the main repository.

## 📜 License

MIT License - Part of Artifact Engine