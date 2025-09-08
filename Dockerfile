FROM continuumio/miniconda3:latest
RUN conda update conda
RUN conda install python=3.10.18

## Create the environment:
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

# Create application directory
WORKDIR /app

# Copy your script and any other necessary files
# Assuming your script is in sybil/ subfolder on host
COPY . .

# Create mount points for external data and output
# These will be empty in the image but used as mount points
RUN mkdir -p data output
ENTRYPOINT ["python3", "lungmask_BIDS.py"]
