# Installation Guide

This guide will help you install the RAG Sample application properly.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Groq API key (get one at https://console.groq.com/)

## Installation Methods

### Method 1: Using the Setup Script (Recommended)

1. **Run the setup script:**
   ```bash
   ./setup_venv.sh
   ```

2. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```

3. **Install the package in development mode:**
   ```bash
   pip install -e .
   ```

4. **Configure your API key:**
   ```bash
   cp env.example .env
   # Edit .env and add your GROQ_API_KEY
   ```

5. **Run the application:**
   ```bash
   rag-sample
   ```

### Method 2: Manual Installation

1. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install the package in development mode:**
   ```bash
   pip install -e .
   ```

4. **Configure your environment:**
   ```bash
   cp env.example .env
   # Edit .env and add your GROQ_API_KEY
   ```

5. **Run the application:**
   ```bash
   rag-sample
   # Or: python -m rag_sample.cli
   ```

### Method 3: Using pipx (Alternative)

If you prefer to use pipx for isolated installations:

1. **Install pipx:**
   ```bash
   brew install pipx  # On macOS
   # Or: pip install pipx
   ```

2. **Install the package:**
   ```bash
   pipx install -e .
   ```

3. **Run the application:**
   ```bash
   rag-sample
   ```

## Troubleshooting

### Error: "externally-managed-environment"

This error occurs on macOS with Homebrew Python. Solutions:

1. **Use a virtual environment (Recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -e .
   ```

2. **Use pipx:**
   ```bash
   brew install pipx
   pipx install -e .
   ```

3. **Override the restriction (Not recommended):**
   ```bash
   pip install -e . --break-system-packages
   ```

### Error: "command not found: pip"

Try using:
```bash
python3 -m pip install -e .
```

### Error: "No module named 'setuptools'"

Install setuptools:
```bash
pip install setuptools wheel
```

### Error: "Permission denied"

Use a virtual environment or install with --user:
```bash
pip install -e . --user
```

## Verification

After installation, verify everything works:

1. **Check installation:**
   ```bash
   rag-sample --help
   ```

2. **Test with a simple question:**
   ```bash
   rag-sample
   # Then ask: "Hello, how are you?"
   ```

## Development Setup

For development work:

1. **Install in development mode:**
   ```bash
   pip install -e .
   ```

2. **Install development dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install pytest black flake8 mypy
   ```

3. **Run tests:**
   ```bash
   pytest
   ```

4. **Format code:**
   ```bash
   black src/
   ```

5. **Lint code:**
   ```bash
   flake8 src/
   ```

## Uninstallation

To uninstall the package:

```bash
pip uninstall rag-sample
```

Or if installed with pipx:
```bash
pipx uninstall rag-sample
```
