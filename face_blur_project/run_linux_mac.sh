#!/bin/bash
echo "==================================================="
echo "  Face Blur Project - Auto Setup and Run Script"
echo "==================================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed or not in your PATH."
    exit 1
fi

# Step 1: Create Virtual Environment
if [ ! -d "venv" ]; then
    echo "[INFO] Creating virtual environment..."
    python3 -m venv venv
fi

# Step 2: Activate Virtual Environment
echo "[INFO] Activating virtual environment..."
source venv/bin/activate

# Step 3: Install Dependencies
echo "[INFO] Installing requirements (this may take a moment)..."
pip install -r requirements.txt --quiet

# Step 4: Download and Trim Sample Datasets
if [ ! -f "sample_inputs/sample_image.jpg" ]; then
    echo "[INFO] Downloading sample datasets..."
    python scripts/download_dataset.py
    echo "[INFO] Trimming sample videos to 10 seconds..."
    python scripts/trim_videos.py
fi

# Step 5: Run the Web UI
echo "==================================================="
echo "[INFO] Starting the Streamlit Web Interface..."
echo "==================================================="
streamlit run streamlit_app.py
