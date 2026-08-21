#!/usr/bin/env python3
"""
🚀 Example: Training a Model with Artifact Engine
Demonstrates the complete training pipeline with live telemetry
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from training import (
    ArtifactTrainingEngine,
    TrainingConfig,
    DataStreamer,
    DataSource
)

async def example_basic_training():
    """Example: Basic model training"""
    print("\n" + "="*60)
    print("📚 Example 1: Basic Training")
    print("="*60 + "\n")
    
    # Configure training
    config = TrainingConfig(
        model_name="example-gpt",
        model_size="7B",
        batch_size=32,
        learning_rate=3e-4,
        epochs=3,
        output_dir="./examples/artifacts"
    )
    
    # Create training engine
    engine = ArtifactTrainingEngine(config)
    
    # Run complete pipeline
    sources = [
        "https://huggingface.co/datasets/c4/en/val/0000.json"
    ]
    
    results = await engine.full_pipeline(sources)
    
    print("\n✅ Training complete!")
    return results

async def example_data_streaming():
    """Example: Smart data streaming from multiple sources"""
    print("\n" + "="*60)
    print("🌊 Example 2: Data Streaming")
    print("="*60 + "\n")
    
    # Initialize data streamer
    streamer = DataStreamer()
    
    # Stream from different sources
    sources = [
        DataSource(
            url="https://api.example.com/data",
            type="api",
            format="json",
            headers={"Authorization": "Bearer token"}
        ),
        DataSource(
            url="https://huggingface.co/dataset.json",
            type="url",
            format="json"
        ),
        DataSource(
            url="./local_data.csv",
            type="file",
            format="csv"
        )
    ]
    
    async with streamer as s:
        for source in sources:
            try:
                if source.type == "api":
                    data = await s.stream_from_api(source.url, headers=source.headers)
                elif source.type == "url":
                    data = await s.stream_from_url(source.url)
                elif source.type == "file":
                    data = await s.stream_from_file(source.url)
                
                print(f"✅ Streamed from {source.url}")
                print(f"   Type: {data.detected_type}")
                print(f"   Size: {data.size:,} bytes")
                print(f"   Quality: {data.quality_score:.2%}")
                
            except Exception as e:
                print(f"❌ Failed to stream from {source.url}: {e}")
    
    print("\n✅ Data streaming complete!")

async def example_custom_training():
    """Example: Custom training configuration"""
    print("\n" + "="*60)
    print("⚙️ Example 3: Custom Training Configuration")
    print("="*60 + "\n")
    
    # Advanced configuration
    config = TrainingConfig(
        model_name="custom-gpt",
        model_size="13B",
        batch_size=64,
        learning_rate=2e-4,
        epochs=5,
        warmup_steps=500,
        max_seq_length=4096,
        gradient_accumulation_steps=8,
        mixed_precision="bf16",
        use_8bit=True,
        lora_r=16,
        lora_alpha=32,
        output_dir="./examples/custom_artifacts"
    )
    
    print("📝 Configuration:")
    print(f"   Model: {config.model_name} ({config.model_size})")
    print(f"   Batch Size: {config.batch_size}")
    print(f"   Learning Rate: {config.learning_rate}")
    print(f"   Epochs: {config.epochs}")
    print(f"   Mixed Precision: {config.mixed_precision}")
    print(f"   8-bit: {config.use_8bit}")
    print(f"   LoRA: r={config.lora_r}, alpha={config.lora_alpha}")
    
    engine = ArtifactTrainingEngine(config)
    
    # Run training
    results = await engine.train()
    
    print("\n✅ Custom training complete!")
    return results

async def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("🚀 ARTIFACT ENGINE TRAINING EXAMPLES")
    print("="*60)
    
    try:
        # Example 1: Basic training
        await example_basic_training()
        
        # Example 2: Data streaming
        await example_data_streaming()
        
        # Example 3: Custom training
        await example_custom_training()
        
        print("\n" + "="*60)
        print("✅ ALL EXAMPLES COMPLETE!")
        print("="*60 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())