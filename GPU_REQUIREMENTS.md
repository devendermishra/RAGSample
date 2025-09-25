# GPU Requirements

## Overview

This RAG (Retrieval-Augmented Generation) application is designed to run efficiently on both CPU and GPU environments. While the application can function without GPU acceleration, using a GPU can significantly improve performance for certain operations.

## GPU Support

### Supported Operations
- **Embedding Generation**: HuggingFace sentence transformers can utilize GPU for faster embedding computation
- **LLM Inference**: Groq API handles GPU acceleration on their servers (no local GPU required for LLM calls)

### Local GPU Requirements (Optional)

#### Minimum Requirements
- **GPU**: NVIDIA GPU with CUDA support (GTX 1060 or better recommended)
- **VRAM**: 4GB minimum, 8GB recommended for optimal performance
- **CUDA**: Version 11.8 or higher
- **Driver**: Latest NVIDIA drivers

#### Recommended Configuration
- **GPU**: RTX 3080/4080 or better
- **VRAM**: 12GB or more
- **CUDA**: Version 12.0 or higher

### Installation with GPU Support

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install additional GPU-accelerated packages
pip install accelerate
pip install bitsandbytes  # For 8-bit quantization (optional)
```

### Environment Variables for GPU

Add these to your `.env` file for GPU optimization:

```bash
# GPU Configuration
CUDA_VISIBLE_DEVICES=0  # Use first GPU
TOKENIZERS_PARALLELISM=true
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### Performance Considerations

#### CPU vs GPU Performance
- **CPU Only**: Suitable for development and small-scale usage
- **GPU Accelerated**: Recommended for production and large document collections

#### Memory Usage
- **CPU Mode**: ~2-4GB RAM usage
- **GPU Mode**: ~4-8GB VRAM + 2-4GB RAM usage

### Cloud GPU Options

#### Groq API (Recommended)
- **No local GPU required**
- **High-performance inference** on Groq's optimized hardware
- **Pay-per-use pricing**
- **Supports multiple models** (Llama, Mixtral, etc.)

#### Alternative Cloud Providers
- **Google Colab**: Free GPU access (T4, limited hours)
- **AWS EC2**: G4 instances with T4 GPUs
- **Azure**: NC-series instances
- **Lambda Labs**: Affordable GPU instances

### Troubleshooting GPU Issues

#### Common Issues
1. **CUDA out of memory**: Reduce batch size or use CPU fallback
2. **Driver compatibility**: Update NVIDIA drivers
3. **CUDA version mismatch**: Ensure PyTorch CUDA version matches system

#### Fallback to CPU
The application automatically falls back to CPU if GPU is not available:

```python
# Automatic fallback in embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
)
```

### Monitoring GPU Usage

```bash
# Monitor GPU usage
nvidia-smi

# Monitor memory usage
watch -n 1 nvidia-smi
```

### Cost Optimization

#### For Development
- Use CPU-only mode for testing
- Use Groq API for LLM inference (no local GPU needed)

#### For Production
- Consider GPU instances only for high-volume embedding generation
- Use Groq API for LLM calls to avoid local GPU costs
- Implement caching to reduce repeated computations

## Conclusion

While GPU acceleration can improve performance, this RAG application is designed to work efficiently on CPU-only systems. The Groq API handles GPU acceleration for LLM inference, making local GPU requirements minimal for most use cases.
