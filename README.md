# Personal documentation
> Check [README_original](README_original.md) for further documentation and citation!

- Requirements have been edited to safely set up future environments and docker image.
- Added `lungmask_BIDS.py`: This script performs automated lung segmentation on CT images and optionally computes emphysema metrics. It supports both lung-level and lobe-level segmentation and can process multiple images in batch mode following BIDS (Brain Imaging Data Structure) naming conventions.

## Installation (via github and pip)
```shell 
git clone https://github.com/giurdeluca/lungmask.git
cd lungmask
conda create env -n python=3.10 <env_name>
conda activate <env_name>
pip install -r requirements.txt
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

## Run inference with lungmask_BIDS.py
### Basic Usage

```bash
python lungmask_script.py --input-list file_paths.txt --output-dir ./results/
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--segmentation` | str | `lungs` | Segmentation type: `lungs` (left/right) or `lobes` (5 lobes) |
| `--fill` | flag | False | Run parallel fill model for improved segmentation (requires more resources) |
| `--input-list` | str | `file_paths.txt` | Path to text file containing input file paths |
| `--output-dir` | str | `./derived/pipeline/` | Directory to save output files |
| `--emphysema` | flag | False | Compute emphysema metrics on segmented lungs |

### Examples

#### Basic lung segmentation:
```bash
python lungmask_script.py --segmentation lungs --input-list my_files.txt
```

#### Lobe segmentation with fill model:
```bash
python lungmask_script.py --segmentation lobes --fill --input-list my_files.txt
```

#### Full analysis with emphysema metrics:
```bash
python lungmask_script.py --segmentation lungs --emphysema --input-list my_files.txt --output-dir ./analysis_results/
```

## Input Format

### File List Structure

Create a text file (e.g., `file_paths.txt`) with one CT scan path per line:

```
/path/to/sub-001/ses-baseline/ct/sub-001_ses-baseline_ct.nii.gz
/path/to/sub-002/ses-baseline/ct/sub-002_ses-baseline_ct.nii.gz
/path/to/sub-003/ses-followup/ct/sub-003_ses-followup_ct.nii.gz
```

### BIDS Naming Convention

Input files must follow BIDS naming convention with `sub-` and `ses-` identifiers:
- `sub-XXX`: Subject identifier
- `ses-YYY`: Session identifier
- File format: `sub-XXX_ses-YYY_ct.nii.gz`

## Output Structure

The script generates outputs following BIDS structure:

```
output_dir/
├── sub-001/
│   └── ses-baseline/
│       └── ct/
│           ├── sub-001_ses-baseline_desc-lungsmask.nii.gz      # Lung mask
│           ├── sub-001_ses-baseline_desc-emph.txt              # Emphysema scores (if --emphysema)
│           └── sub-001_ses-baseline_desc-laa950mask.nii.gz     # LAA950 mask (if --emphysema)
├── emphysema_results.csv                                       # Summary CSV (if --emphysema)
└── lung_mask.log                                              # Processing log
```

## Emphysema Metrics

When `--emphysema` flag is used, the following metrics are computed:

### Low Attenuation Areas (LAA)
- **LAA950**: Percentage of lung voxels ≤ -950 HU
- **LAA910**: Percentage of lung voxels ≤ -910 HU  
- **LAA856**: Percentage of lung voxels ≤ -856 HU

### High Attenuation Areas (HAA)
- **HAA700**: Percentage of lung voxels ≥ -700 HU
- **HAA600**: Percentage of lung voxels ≥ -600 HU
- **HAA500**: Percentage of lung voxels ≥ -500 HU
- **HAA250**: Percentage of lung voxels ≥ -250 HU

### Statistical Measures
- **Percentiles**: 10th and 15th percentiles of HU values
- **HU Statistics**: Mean, standard deviation, median, kurtosis, skewness, min, max

### Output Files (Emphysema Mode)

1. **Individual Text Files** (`*_desc-emph.txt`): Human-readable emphysema scores
2. **LAA950 Masks** (`*_desc-laa950mask.nii.gz`): Binary masks of emphysematous regions
3. **Summary CSV** (`emphysema_results.csv`): All metrics for all processed images

## Models Used

- **R231**: Default model for lung segmentation
- **LTRCLobes**: Model for lobe-level segmentation
- **Fill Model**: Optional model for improved segmentation quality

## Logging

The script provides comprehensive logging:
- **Console Output**: Real-time processing updates
- **Log File**: Detailed processing log saved to `lung_mask.log`
- **Error Handling**: Graceful handling of processing failures

## Run docker image

1. Build docker image (18 GB)
```shell
docker build -t lungmask .
```

2. Run the container with GPU support
> **Important**: Remember to change data paths in the input list txt file to match the docker container paths (e.g., `/app/data/...`)!

```shell
docker run --gpus all --rm \
  -v /local/path/to/BIDS/folder:/app/data \
  -v /local/path/to/results/folder:/app/output \
  -v /local/path/to/input-list.txt:/app/data/input-list.txt:ro \
  lungmask \
  --segmentation lobes \
  --input-list /app/data/input-list.txt \
  --output-dir /app/output \
  --emphysema
```

### Docker Notes
- **GPU Support**: `--gpus all` enables GPU acceleration for faster processing
- **File Paths**: Update your `input-list.txt` to use container paths (e.g., `/app/data/sub-001/ses-baseline/ct/sub-001_ses-baseline_ct.nii.gz`)
- **Permissions**: Ensure the local output directory has appropriate write permissions
- **Image Size**: The Docker image is approximately 18 GB due to deep learning model dependencies