#!/bin/bash
# 🚀 Setup Script for Artifact Training Module

echo "🧠 Setting up Artifact Training Module..."
echo ""

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install rich pandas aiohttp

# Make CLI executable
echo "🔧 Making CLI executable..."
chmod +x /home/adam/Projects/ARC/artifact-engine/artifact
chmod +x /home/adam/Projects/ARC/artifact-engine/examples/train_example.py

# Create directories
echo "📁 Creating directories..."
mkdir -p /home/adam/Projects/ARC/artifact-engine/artifacts
mkdir -p /home/adam/Projects/ARC/artifact-engine/data
mkdir -p /home/adam/Projects/ARC/artifact-engine/logs

# Test installation
echo ""
echo "✅ Installation complete!"
echo ""
echo "🚀 Quick Start:"
echo "   ./artifact --help"
echo "   ./artifact train --help"
echo "   python examples/train_example.py"
echo ""
echo "📚 Documentation:"
echo "   src/training/README.md"
echo "   TRAINING_MODULE_SUMMARY.md"
echo ""