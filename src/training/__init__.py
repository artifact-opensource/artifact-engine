#!/usr/bin/env python3
"""
🧠 Artifact Training Module
Build GPT from ground up with live telemetry
"""

from .trainer import (
    ArtifactTrainingEngine,
    TrainingPipeline,
    train_command,
    build_command,
    serve_command
)

from .terminal_ui import (
    ArtifactTrainer,
    TrainingConfig,
    TrainingMetrics,
    TrainingPhase
)

from .data_streamer import (
    DataStreamer,
    DataSource,
    DataStream
)

__all__ = [
    'ArtifactTrainingEngine',
    'TrainingPipeline',
    'train_command',
    'build_command',
    'serve_command',
    'ArtifactTrainer',
    'TrainingConfig',
    'TrainingMetrics',
    'TrainingPhase',
    'DataStreamer',
    'DataSource',
    'DataStream'
]

__version__ = "1.0.0"
__author__ = "Artifact Virtual"
__description__ = "Build GPT from ground up • Live telemetry • Smart data ingestion"