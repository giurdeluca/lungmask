from lungmask import LMInferer
import SimpleITK as sitk
from scipy.stats import kurtosis, skew
import argparse
import sys
import os
import time
import logging
import re
import numpy as np
import csv

parser = argparse.ArgumentParser(description='LungMask segmentation')
parser.add_argument('--segmentation', default='lungs', type=str, choices=['lungs','lobes'],
                    help='segmentation: lungs if left and right lung, lobes if all 5 lobes. Default: lungs')
parser.add_argument('--fill', action='store_true', help='wheter to run a parallel fill model to improve the segmentation. Be careful: it requires more time and resources!')
parser.add_argument('--input-list', default='file_paths.txt', type=str,
                    help='input-list: path to the text file containing the input file paths. Default: ./file_paths.txt')
parser.add_argument('--output-dir', default='./derived/pipeline/', type=str,
                    help='output-dir: directory to save output files. Default: derived/pipeline/')
parser.add_argument('--emphysema', action='store_true', help='wheter to compute emphysema metrics (LAA950 etc) on smooth images.')
opt = parser.parse_args()

def create_inferer(segmentation, fill):
    modelname = 'R231' if segmentation == 'lungs' else "LTRCLobes"
    if fill:
        inferer = LMInferer(modelname=modelname, fillmodel='R231')
    else:
        inferer = LMInferer(modelname=modelname)
    return inferer

def write_scores_to_text(scores, file_path):
    """Write emphysema scores to individual text file in human-readable format."""
    with open(file_path, 'w') as output_file:
        output_file.write("EMPHYSEMA ANALYSIS RESULTS\n")
        output_file.write("=" * 40 + "\n\n")
        
        # LAA scores section
        output_file.write("Low Attenuation Areas (LAA):\n")
        output_file.write(f"  LAA950: {scores.get('LAA950', 'N/A'):.6f}\n")
        output_file.write(f"  LAA910: {scores.get('LAA910', 'N/A'):.6f}\n")
        output_file.write(f"  LAA856: {scores.get('LAA856', 'N/A'):.6f}\n\n")
        
        # HAA scores section
        output_file.write("High Attenuation Areas (HAA):\n")
        output_file.write(f"  HAA700: {scores.get('HAA700', 'N/A'):.6f}\n")
        output_file.write(f"  HAA600: {scores.get('HAA600', 'N/A'):.6f}\n")
        output_file.write(f"  HAA500: {scores.get('HAA500', 'N/A'):.6f}\n")
        output_file.write(f"  HAA250: {scores.get('HAA250', 'N/A'):.6f}\n\n")
        
        # Percentiles section
        output_file.write("Percentiles:\n")
        output_file.write(f"  15thPercentile: {scores.get('Perc15', 'N/A'):.2f}\n")
        output_file.write(f"  10thPercentile: {scores.get('Perc10', 'N/A'):.2f}\n\n")
        
        # HU statistics section
        output_file.write("Hounsfield Unit Statistics:\n")
        output_file.write(f"  Mean: {scores.get('HUMean', 'N/A'):.2f}\n")
        output_file.write(f"  Std: {scores.get('HUStd', 'N/A'):.2f}\n")
        output_file.write(f"  Median: {scores.get('HUMedian', 'N/A'):.2f}\n")
        output_file.write(f"  Kurtosis: {scores.get('HUKurtosis', 'N/A'):.4f}\n")
        output_file.write(f"  Skewness: {scores.get('HUSkewness', 'N/A'):.4f}\n")
        output_file.write(f"  Min: {scores.get('HUMin', 'N/A'):.2f}\n")
        output_file.write(f"  Max: {scores.get('HUMax', 'N/A'):.2f}\n")

def append_to_csv(csv_path, data_row):
    """Append a single row of data to the CSV file."""
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data_row)

def compute_emphysema(image_sitk, mask_array, save=False):
    scores = {}
    try:
        image_array = sitk.GetArrayFromImage(image_sitk)
        mask_array_bool = mask_array.astype(bool)
        mask_array_bool_sum = np.sum(mask_array_bool)
        lung_array = image_array[mask_array_bool]

        if mask_array_bool_sum > 0:
            # Compute LAA scores
            scores['LAA950'] = float(np.sum(lung_array <= -950.)) / mask_array_bool_sum
            scores['LAA910'] = float(np.sum(lung_array <= -910.)) / mask_array_bool_sum
            scores['LAA856'] = float(np.sum(lung_array <= -856.)) / mask_array_bool_sum

            # Compute HAA scores
            scores['HAA700'] = float(np.sum(lung_array >= -700.)) / mask_array_bool_sum
            scores['HAA600'] = float(np.sum(lung_array >= -600.)) / mask_array_bool_sum
            scores['HAA500'] = float(np.sum(lung_array >= -500.)) / mask_array_bool_sum
            scores['HAA250'] = float(np.sum(lung_array >= -250.)) / mask_array_bool_sum

            # Compute percentiles
            scores['Perc15'] = np.percentile(lung_array, 15)
            scores['Perc10'] = np.percentile(lung_array, 10)

            # Compute HU statistics
            scores['HUMean'] = np.mean(lung_array)
            scores['HUStd'] = np.std(lung_array)
            scores['HUKurtosis'] = kurtosis(lung_array, bias=False, fisher=True)
            scores['HUSkewness'] = skew(lung_array, bias=False)
            scores['HUMedian'] = np.median(lung_array)
            scores['HUMin'] = np.min(lung_array)
            scores['HUMax'] = np.max(lung_array)

            # Create boolean emphysema mask based on LAA950 threshold
            if save:
                laa950_mask = np.zeros_like(image_array, dtype=np.uint8)
                # Create emphysema condition: lung areas AND below -950 HU
                emphysema_condition = mask_array_bool & (image_array <= -950)
                laa950_mask[emphysema_condition] = 1
                laa950_mask_sitk = sitk.GetImageFromArray(laa950_mask)
                laa950_mask_sitk.CopyInformation(image_sitk)
                laa950_mask_sitk = sitk.Cast(laa950_mask_sitk, sitk.sitkUInt8)
                return scores, laa950_mask_sitk
            else:
                return scores
    except Exception as e:
        print(f"FAILED EMPHYSEMA ESTIMATION {e}")
        return None
    
# Create outputdir
os.makedirs(opt.output_dir, exist_ok=True)
# Define the log file path
log_file_path = os.path.join(opt.output_dir, 'lung_mask.log')
if os.path.exists(log_file_path):
    os.remove(log_file_path)
    open(log_file_path, 'w').close()

# Define CSV file path and headers
csv_file_path = os.path.join(opt.output_dir, 'emphysema_results.csv')
csv_headers = [
    'input_path', 'status', 'processing_time_seconds',
    # LAA scores
    'LAA950', 'LAA910', 'LAA856',
    # HAA scores  
    'HAA700', 'HAA600', 'HAA500', 'HAA250',
    # Percentiles
    'Perc15', 'Perc10',
    # HU statistics
    'HUMean', 'HUStd', 'HUKurtosis', 'HUSkewness', 'HUMedian', 'HUMin', 'HUMax'
]
# Initialize CSV with headers (only if emphysema analysis is requested)
if opt.emphysema:
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_headers)

# Configure logging for both console and file separately
logger = logging.getLogger('lung_mask')
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Log parsed arguments
logger.info(f'Parsed Arguments: {opt}')

# main
inferer = create_inferer(opt.segmentation, opt.fill)
with open(opt.input_list, 'r') as file_paths:
    for line in file_paths:
        # Define regular expressions to match 'sub' and 'ses' parts
        sub_pattern = r'sub-\d+'
        ses_pattern = r'ses-[A-Za-z0-9]+'

        input_path = line.strip()
        logger.info(f'input path: {input_path}')
        
        # Extract 'sub' and 'ses' parts from the file path
        sub_match = re.search(sub_pattern, input_path)
        ses_match = re.search(ses_pattern, input_path)
        if not sub_match or not ses_match:
            logger.error(f'BIDS format not found in {input_path}')
            continue
            
        sub_part = sub_match.group(0)
        ses_part = ses_match.group(0)
        
        # output files must be stored respecting BIDS structure
        output_path = os.path.join(opt.output_dir, sub_part, ses_part, 'ct')
        os.makedirs(output_path, exist_ok=True)
        file_name = os.path.basename(input_path)
        lungs_file_name = file_name.replace('ct.nii.gz', f'desc-{opt.segmentation}mask.nii.gz')
        lungs_file_path = os.path.join(output_path, lungs_file_name)
        
        start_time = time.time()
        
        # Initialize default values for CSV row (when emphysema analysis is enabled)
        status = "fail"
        scores = {}
        
        try:
            image = sitk.ReadImage(input_path)
            mask_array = inferer.apply(image)
            mask = sitk.GetImageFromArray(mask_array)
            mask.CopyInformation(image)
            mask = sitk.Cast(mask, sitk.sitkUInt8)
            sitk.WriteImage(mask, lungs_file_path)
            
            # If emphysema analysis is requested
            if opt.emphysema:
                score_file_name = file_name.replace('ct.nii.gz', 'desc-emph.txt')
                score_file_path = os.path.join(output_path, score_file_name)
                laa950_file_name = file_name.replace('ct.nii.gz', f'desc-laa950mask.nii.gz')
                laa950_file_path = os.path.join(output_path, laa950_file_name)
                try:
                    scores, laa950_mask = compute_emphysema(image, mask_array, save=True)
                    
                    if scores is not None:
                        # Write individual text file
                        write_scores_to_text(scores, score_file_path)
                        sitk.WriteImage(laa950_mask, laa950_file_path)
                        status = "success"
                        logger.info(f'Emphysema scores computed and saved to {score_file_path}')
                        logger.info(f'LAA950 mask saved to {laa950_file_path}')
                    else:
                        # Write failure message to text file
                        with open(score_file_path, 'w') as output_file:
                            output_file.write('FAILED EMPHYSEMA ESTIMATION\n')
                        logger.error(f'Failed to compute emphysema scores for {input_path}')
                        
                except Exception as e:
                    logger.error(f'Error computing emphysema scores for {input_path}: {e}')
                    # Write failure message to text file
                    score_file_path = os.path.join(output_path, score_file_name)
                    with open(score_file_path, 'w') as output_file:
                        output_file.write(f'FAILED EMPHYSEMA ESTIMATION: {e}\n')
            else:
                status = "success"  # Lung mask was successful even without emphysema
                
        except RuntimeError as e:
            logger.error(f'Error processing {input_path}: {e}')
            continue

        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Write to CSV (only if emphysema analysis was requested)
        if opt.emphysema:
            # Prepare CSV row with all values
            row = [
                input_path, sub_part, ses_part, status, elapsed_time,
                # LAA scores
                scores.get('LAA950', 'N/A'), scores.get('LAA910', 'N/A'), scores.get('LAA856', 'N/A'),
                # HAA scores
                scores.get('HAA700', 'N/A'), scores.get('HAA600', 'N/A'), 
                scores.get('HAA500', 'N/A'), scores.get('HAA250', 'N/A'),
                # Percentiles
                scores.get('Perc15', 'N/A'), scores.get('Perc10', 'N/A'),
                # HU statistics
                scores.get('HUMean', 'N/A'), scores.get('HUStd', 'N/A'), 
                scores.get('HUKurtosis', 'N/A'), scores.get('HUSkewness', 'N/A'),
                scores.get('HUMedian', 'N/A'), scores.get('HUMin', 'N/A'), scores.get('HUMax', 'N/A')
            ]
            append_to_csv(csv_file_path, row)

        logger.info(f'Processed {input_path} in {elapsed_time:.2f} seconds. Mask saved to {lungs_file_path}.')

logger.info(f'Log file saved to: {log_file_path}')
if opt.emphysema:
    logger.info(f'Emphysema results CSV saved to: {csv_file_path}')