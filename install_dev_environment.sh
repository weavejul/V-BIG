#!/bin/bash

# Development environment setup script for V-BIG package

echo "Setting up V-BIG development environment..."

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "Virtual environment detected: $VIRTUAL_ENV"
else
    echo "Warning: No virtual environment detected. Consider creating one:"
    echo "  python -m venv vbig-env"
    echo "  source vbig-env/bin/activate  # On Windows: vbig-env\\Scripts\\activate"
    echo ""
fi

# Install the package in development mode
echo "Installing V-BIG package in development mode..."
pip install -e .

# Install development dependencies
echo "Installing development dependencies..."
pip install -e ".[dev,wandb,viz]"

# Download NLTK data
echo "Downloading NLTK stopwords..."
python -c "import nltk; nltk.download('stopwords', quiet=True)"

# Test installation
echo "Testing package installation..."
python -c "
try:
    import vbig
    print('✅ V-BIG package imported successfully!')
    
    # Test key imports
    from vbig.attribution import compute_ig_attributions
    from vbig.training import IGTrainer
    from vbig.data import NLIDataProcessor
    from vbig.visualization import visualize_attributions
    print('✅ All key modules imported successfully!')
    
    print('📦 Package version:', vbig.__version__)
    
except ImportError as e:
    print('❌ Import error:', e)
    exit(1)
"

echo ""
echo "🎉 Setup complete! You can now:"
echo "  1. Train models: python examples/train_vbig_model.py --help"
echo "  2. Analyze attributions: python examples/analyze_attributions.py --help"  
echo "  3. Compare models: python examples/compare_models.py --help"
echo "  4. Use migration helper: python migrate_from_old_structure.py --all"
echo ""
echo "For help migrating from old code structure:"
echo "  python migrate_from_old_structure.py --all"