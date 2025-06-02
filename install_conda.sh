#!/bin/bash
cd /home/foresti

CONDA_DIR="/home/foresti/miniconda"

# Function to check if conda is working properly
check_conda_working() {
    if [ -f "$CONDA_DIR/bin/conda" ]; then
        # Try to run conda --version to see if it works
        "$CONDA_DIR/bin/conda" --version >/dev/null 2>&1
        return $?
    else
        return 1
    fi
}

# Check if conda directory exists but is broken
if [ -d "$CONDA_DIR" ]; then
    echo "Found existing conda directory at $CONDA_DIR"
    
    if check_conda_working; then
        echo "Conda is working properly. Updating..."
        export PATH="$CONDA_DIR/bin:$PATH"
        conda update -n base -c defaults conda -y
    else
        echo "Conda installation appears corrupted. Removing and reinstalling..."
        rm -rf "$CONDA_DIR"
        
        # Fresh installation
        echo "Installing fresh conda..."
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /home/foresti/miniconda.sh
        bash /home/foresti/miniconda.sh -b -p "$CONDA_DIR"
        
        # Initialize conda for bash
        "$CONDA_DIR/bin/conda" init bash
        
        # Add to current session PATH
        export PATH="$CONDA_DIR/bin:$PATH"
    fi
else
    echo "No existing conda installation found. Installing fresh..."
    
    # Download and install miniconda
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /home/foresti/miniconda.sh
    bash /home/foresti/miniconda.sh -b -p "$CONDA_DIR"
    
    # Initialize conda for bash
    "$CONDA_DIR/bin/conda" init bash
    
    # Add to current session PATH
    export PATH="$CONDA_DIR/bin:$PATH"
fi

# Verify installation
echo "Verifying conda installation..."
if check_conda_working; then
    echo "Conda version: $("$CONDA_DIR/bin/conda" --version)"
    echo "Conda installation/update complete!"
else
    echo "ERROR: Conda installation failed or is still corrupted."
    exit 1
fi

echo "Please restart your terminal or run 'source ~/.bashrc' to ensure conda is activated."