import os, sys, argparse, yaml, math
import numpy as np
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from math import ceil 
import wfdb
from imgaug import augmenters as iaa

BIT_NAN_16 = -(2.**15)

def read_config_file(config_file):
    """Read YAML config file

    Args:
        config_file (str): Complete path to the config file
    
    Returns:
        configs (dict): Returns dictionary with all the configs
    """
    with open(config_file) as f:
            yamlObject = yaml.safe_load(f)

    args = dict()
    for key in yamlObject:
        args[key] = yamlObject[key]

    return args

def find_records(folder, output_dir):
    header_files = list()
    recording_files = list()

    for root, directories, files in os.walk(folder):
        files = sorted(files)
        for file in files:
            extension = os.path.splitext(file)[1]
            if extension == '.mat':
                record = os.path.relpath(os.path.join(root, file.split('.')[0] + '.mat'), folder)
                hd = os.path.relpath(os.path.join(root, file.split('.')[0] + '.hea'), folder)
                recording_files.append(record)
                header_files.append(hd)
            if extension == '.dat':
                record = os.path.relpath(os.path.join(root, file.split('.')[0] + '.dat'), folder)
                hd = os.path.relpath(os.path.join(root, file.split('.')[0] + '.hea'), folder)
                header_files.append(hd)
                recording_files.append(record)
    
    if recording_files == []:
        raise Exception("The input directory does not have any WFDB compatible ECG files, please re-check the folder!")


    for file in recording_files:
        f, ext = os.path.splitext(file)
        f1 = f.split('/')[:-1]
        f1 = '/'.join(f1)

        if os.path.exists(os.path.join(output_dir, f1)) == False:
            os.makedirs(os.path.join(output_dir, f1))

    return header_files, recording_files


def find_files(data_directory):
    header_files = list()
    recording_files = list()

    for f in sorted(os.listdir(data_directory)):

        if(os.path.isdir(os.path.join(data_directory, f))):
            
            for file in sorted(os.listdir(os.path.join(data_directory,f))):
                root, extension = os.path.splitext(file)
                
                if not root.startswith('.'):

                    if extension=='.mat':
                        header_file = os.path.join(os.path.join(data_directory,f), root + '.hea')
                        recording_file = os.path.join(os.path.join(data_directory,f), root + '.mat')

                        if os.path.isfile(header_file) and os.path.isfile(recording_file):
                            header_files.append(header_file)
                            recording_files.append(recording_file)

                    if extension=='.dat':
                        header_file = os.path.join(os.path.join(data_directory,f), root + '.hea')
                        recording_file = os.path.join(os.path.join(data_directory,f), root + '.dat')

                        if os.path.isfile(header_file) and os.path.isfile(recording_file):
                            header_files.append(header_file)
                            recording_files.append(recording_file)
                            
        else:

            root, extension = os.path.splitext(f)

            if not root.startswith('.'):
                #Based on the recording format, we save the file names differently
                if extension=='.mat':
                    header_file = os.path.join(data_directory, root + '.hea')
                    recording_file = os.path.join(data_directory, root + '.mat')
                    if os.path.isfile(header_file) and os.path.isfile(recording_file):
                        header_files.append(header_file)
                        recording_files.append(recording_file)

                if extension=='.dat':
                    header_file = os.path.join(data_directory, root + '.hea')
                    recording_file = os.path.join(data_directory, root + '.dat')
                    if os.path.isfile(header_file) and os.path.isfile(recording_file):
                        header_files.append(header_file)
                        recording_files.append(recording_file)

    return header_files, recording_files

def load_header(header_file):
    with open(header_file, 'r') as f:
        header = f.read()
    return header

# Load recording file as an array.
def load_recording(recording_file, header=None,key='val'):
    rootname,extension = os.path.splitext(recording_file)
    #Load files differently based on file format
    if extension=='.dat':
        recording = wfdb.rdrecord(rootname)
        return recording.p_signal
    if extension=='.mat':
        recording = loadmat(recording_file)[key]
    return recording

# Get leads from header.
def get_leads(header):
    leads = list()
    for i, l in enumerate(header.split('\n')):
        entries = l.split(' ')
        if i==0:
            num_leads = int(entries[1])
        elif i<=num_leads:
            leads.append(entries[-1])
        else:
            break
    return tuple(leads)

# Get frequency from header.
def get_frequency(header):
    frequency = None
    for i, l in enumerate(header.split('\n')):
        if i==0:
            try:
                frequency = l.split(' ')[2]
                if '/' in frequency:
                    frequency = float(frequency.split('/')[0])
                else:
                    frequency = float(frequency)
            except:
                pass
        else:
            break
    return frequency

# Get analog-to-digital converter (ADC) gains from header.
def get_adc_gains(header, leads):
    adc_gains = np.zeros(len(leads))
    for i, l in enumerate(header.split('\n')):
        entries = l.split(' ')
        if i==0:
            num_leads = int(entries[1])
        elif i<=num_leads:
            current_lead = entries[-1]
            if current_lead in leads:
                j = leads.index(current_lead)
                try:
                    adc_gains[j] = float(entries[2].split('/')[0])
                except:
                    pass
        else:
            break
    return adc_gains


def truncate_signal(signal,sampling_rate,length_in_secs):
    signal=signal[0:int(sampling_rate*length_in_secs)]
    return signal

def create_signal_dictionary(signal,full_leads):
    record_dict = {}
    for k in range(len(full_leads)):
        record_dict[full_leads[k]] = signal[k]
        
    return record_dict

def standardize_leads(full_leads):
    full_leads_array = np.asarray(full_leads)
    
    for i in np.arange(len(full_leads_array)):
        if(full_leads_array[i].upper() not in ('AVR','AVL','AVF')):
            full_leads_array[i] = full_leads_array[i].upper()
        else:
            if(full_leads_array[i].upper()=='AVR'):
                full_leads_array[i] = 'aVR'
            elif(full_leads_array[i].upper()=='AVL'):
                full_leads_array[i] = 'aVL'
            else:
                 full_leads_array[i] = 'aVF'
    return full_leads_array

def rotate_bounding_box(box, origin, angle):
    angle = math.radians(angle)

    transformation = np.ones((2, 2))
    transformation[0][0] = math.cos(angle)
    transformation[0][1] = math.sin(angle)
    transformation[1][0] = -math.sin(angle)
    transformation[1][1] = math.cos(angle)

    new_origin = np.ones((1, 2))
    new_origin[0, 0] = -origin[0]*math.cos(angle) + origin[1]*math.sin(angle)
    new_origin[0, 1] = -origin[0]*math.sin(angle) - origin[1]*math.cos(angle)
    origin = np.reshape(origin, (1, 2))

    transformed_box = np.matmul(box, transformation)    
    transformed_box += origin + new_origin 

    return transformed_box

def read_leads(leads):

    lead_bbs = []
    text_bbs = []
    startTimeStamps = []
    endTimeStamps = []
    labels = []
    plotted_pixels = []
    for i, line in enumerate(leads):
        labels.append(leads[i]['lead_name'])
        st_time_stamp = leads[i]['start_sample']
        startTimeStamps.append(st_time_stamp)
        end_time_stamp = leads[i]['end_sample']
        endTimeStamps.append(end_time_stamp)
        plotted_pixels.append(leads[i]['plotted_pixels'])

        key = "lead_bounding_box"
        if key in leads[i].keys():
            parts = leads[i][key]
            point1 = [parts['0'][0], parts['0'][1]]
            point2 = [parts['1'][0], parts['1'][1]]
            point3 = [parts['2'][0], parts['2'][1]]
            point4 = [parts['3'][0], parts['3'][1]]
            box = [point1, point2, point3, point4]
            lead_bbs.append(box)

        key = "text_bounding_box"
        if key in leads[i].keys():
            parts = leads[i][key]
            point1 = [parts['0'][0], parts['0'][1]]
            point2 = [parts['1'][0], parts['1'][1]]
            point3 = [parts['2'][0], parts['2'][1]]
            point4 = [parts['3'][0], parts['3'][1]]
            box = [point1, point2, point3, point4]
            text_bbs.append(box)

    if len(lead_bbs) != 0:
        lead_bbs = np.array(lead_bbs)
    if len(text_bbs) != 0:
        text_bbs = np.array(text_bbs)

    return lead_bbs, text_bbs, labels, startTimeStamps, endTimeStamps, plotted_pixels

def convert_bounding_boxes_to_dict(lead_bboxes, text_bboxes, labels, startTimeList = None, endTimeList = None, plotted_pixels_dict=None):
    leads_ds = []

    for i in range(len(labels)):
        current_lead_ds = dict()
        if len(lead_bboxes) != 0:
            new_box = dict()
            box = lead_bboxes[i]
            new_box[0] = [round(box[0][0]), round(box[0][1])]
            new_box[1] = [round(box[1][0]), round(box[1][1])]
            new_box[2] = [round(box[2][0]), round(box[2][1])]
            new_box[3] = [round(box[3][0]), round(box[3][1])]
            current_lead_ds["lead_bounding_box"] = new_box

        if len(text_bboxes) != 0:
            new_box = dict()
            box = text_bboxes[i]
            new_box[0] = [round(box[0][0]), round(box[0][1])]
            new_box[1] = [round(box[1][0]), round(box[1][1])]
            new_box[2] = [round(box[2][0]), round(box[2][1])]
            new_box[3] = [round(box[3][0]), round(box[3][1])]
            current_lead_ds["text_bounding_box"] = new_box

        current_lead_ds["lead_name"] = labels[i]
        current_lead_ds["start_sample"] = startTimeList[i]
        current_lead_ds["end_sample"] = endTimeList[i]
        current_lead_ds["plotted_pixels"] = [[plotted_pixels_dict[i][j][0], plotted_pixels_dict[i][j][1]] for j in range(len(plotted_pixels_dict[i]))]
        leads_ds.append(current_lead_ds)

    return leads_ds


def convert_mm_to_volts(mm):
    return float(mm/10)

def convert_mm_to_seconds(mm):
    return float(mm*0.04)

def convert_inches_to_volts(inches):
    return float(inches*2.54)

def convert_inches_to_seconds(inches):
    return float(inches*1.016)

def write_wfdb_file(ecg_frame, filename, rate, header_file, write_dir, full_mode, mask_unplotted_samples):
    # Load original header info
    header_name, extn = os.path.splitext(header_file)
    try:
        header = wfdb.rdheader(header_name)
    except Exception as e:
        raise ValueError(f"Error reading original header file '{header_name}': {e}")

    # Get lead names: Original for output, Standardized for lookup
    original_lead_names = header.sig_name
    # Ensure standardize_leads is available (it should be in this file)
    standardized_lead_names = standardize_leads(list(original_lead_names))

    # --- MODIFICATION 1: Determine sample count (no changes needed here from last version) ---
    if full_mode == 'None':
        lead_keys = list(ecg_frame.keys())
        if not lead_keys:
            raise ValueError("Cannot determine sample count for WFDB header: ecg_frame dictionary is empty when full_mode is None.")
        representative_lead_key = None
        for key in lead_keys:
            if not key.startswith('full'):
                representative_lead_key = key
                break
        if representative_lead_key is None and lead_keys:
             representative_lead_key = lead_keys[0]
        if representative_lead_key is None:
             raise ValueError("Cannot find a representative lead key in ecg_frame when full_mode is None.")
        try:
            samples = len(ecg_frame[representative_lead_key])
        except TypeError:
             raise TypeError(f"Could not determine length for lead '{representative_lead_key}' when full_mode is None.")
        except KeyError:
             raise KeyError(f"Representative key '{representative_lead_key}' selected but not found in ecg_frame when full_mode is None.")
    else:
        intended_key = 'full' + full_mode
        if intended_key not in ecg_frame:
             if full_mode in ecg_frame:
                intended_key = full_mode
                print(f"Warning: Key '{'full' + full_mode}' not found for full_mode lead. Using base key '{full_mode}' for sample count.", file=sys.stderr)
             else:
                raise KeyError(f"Neither '{intended_key}' nor '{full_mode}' found as a key in the ecg_frame dictionary prepared for WFDB writing when full_mode='{full_mode}'.")
        try:
            samples = len(ecg_frame[intended_key])
        except TypeError:
            raise TypeError(f"Could not determine length for lead '{intended_key}' (full_mode='{full_mode}').")
        except KeyError:
             raise KeyError(f"Key '{intended_key}' not found in ecg_frame when trying to determine sample count for full_mode='{full_mode}'.")
    # --- END MODIFICATION 1 ---

    # Prepare the data array
    array = np.zeros((1, samples))
    output_sig_names = [] # Store the ORIGINAL lead names for output

    # --- MODIFICATION 2: Loop using indices and handle keys correctly ---
    for i in range(len(original_lead_names)):
        # Get both original and standardized names for the current index
        lead_name_original = original_lead_names[i]
        lead_name_standardized = standardized_lead_names[i]

        output_sig_names.append(lead_name_original) # Use ORIGINAL name for output sig_name list

        # Determine the correct key to access data from ecg_frame dictionary
        # Use the STANDARDIZED name for logic involving dictionary keys
        current_lead_key = lead_name_standardized # Default to the standardized name
        if full_mode != 'None' and lead_name_standardized == full_mode:
            # If this IS the designated long lead (using standardized name for check)
            # try accessing 'full' + STANDARDIZED name first
            full_key_candidate = 'full' + lead_name_standardized
            if full_key_candidate in ecg_frame:
                current_lead_key = full_key_candidate
            # else: keep current_lead_key as the base standardized name

        # Check if the determined key exists before trying to access it
        if current_lead_key not in ecg_frame:
             available_keys = list(ecg_frame.keys())
             raise KeyError(f"Attempting to access key '{current_lead_key}' (derived from standardized lead '{lead_name_standardized}') for original lead '{lead_name_original}' but key is not in ecg_frame. full_mode='{full_mode}'. Available keys: {available_keys}")

        # Get gain for this lead from the original header using index 'i'
        try:
            adc_gn = header.adc_gain[i]
            if adc_gn == 0:
                 print(f"Warning: ADC gain for lead '{lead_name_original}' is 0. Using default gain of 200.", file=sys.stderr)
                 adc_gn = 200
        except IndexError:
             raise IndexError(f"Could not get ADC gain for lead index {i} ('{lead_name_original}'). Header might be inconsistent.")

        # Access data using the determined key (which is based on standardized name)
        arr = ecg_frame[current_lead_key]
        arr = np.array(arr)

        # Handle NaN values
        nan_mask = np.isnan(arr)
        arr[nan_mask] = BIT_NAN_16 / adc_gn

        # Reshape for concatenation
        arr = arr.reshape((1, arr.shape[0]))

        # Ensure array length matches calculated samples
        if arr.shape[1] != samples:
            raise ValueError(f"Length mismatch for lead '{lead_name_original}' (key: '{current_lead_key}'). Expected {samples}, got {arr.shape[1]}. Check data segmentation logic.")

        # Concatenate the processed lead data
        array = np.concatenate((array, arr), axis=0)
    # --- END MODIFICATION 2 ---

    # Remove the initial dummy row
    array = array[1:]

    # Final checks
    if array.shape[0] != len(output_sig_names):
        raise ValueError(f"Mismatch between number of data rows ({array.shape[0]}) and number of signal names ({len(output_sig_names)}).")
    if array.shape[1] != samples:
         raise ValueError(f"Final WFDB data array has {array.shape[1]} samples, but header expects {samples}.")

    # Get the base name for the output file
    head, tail = os.path.split(filename)
    # Ensure filename used here is the base name intended for the output record
    # Assuming 'filename' passed to write_wfdb_file is like '/path/to/input/00001_lr'
    # or potentially '/path/to/input/00001_lr-0' if processing segments
    # Let's refine how output_record_name is determined based on context
    # From extract_leads.py: `name, ext = os.path.splitext(full_header_file)`
    # `write_wfdb_file(segmented_ecg_data, name, ...)` -> so filename is base name without ext
    # From gen_ecg_image_from_data.py: passes `filename` which is input file path.
    # This seems inconsistent. Let's assume filename passed IS the base record name desired.
    output_record_name = os.path.basename(filename) # Safest approach: take only the file part

    # Write the new WFDB file
    try:
        wfdb.wrsamp(record_name=output_record_name,
                    fs=rate,
                    units=header.units,
                    sig_name=output_sig_names, # Use the ORIGINAL lead names
                    p_signal=array.T,
                    fmt=header.fmt,
                    adc_gain=header.adc_gain,
                    baseline=header.baseline,
                    base_time=header.base_time,
                    base_date=header.base_date,
                    write_dir=write_dir,
                    comments=header.comments)
    except Exception as e:
        # Provide more info on failure
        raise IOError(f"Error writing WFDB file '{os.path.join(write_dir, output_record_name)}' with signals {output_sig_names} and shape {array.T.shape}: {e}")

def get_lead_pixel_coordinate(leads):

    pixel_coordinates = dict()

    for i in range(len(leads)):
        leadName = leads[i]["lead_name"]
        plotted_pixels = np.array(leads[i]["plotted_pixels"])
        pixel_coordinates[leadName] = plotted_pixels

    return pixel_coordinates


def rotate_points(pixel_coordinates, origin, angle):
    rotates_pixel_coords = []
    angle = math.radians(angle)
    transformation = np.ones((2, 2))
    transformation[0][0] = math.cos(angle)
    transformation[0][1] = math.sin(angle)
    transformation[1][0] = -math.sin(angle)
    transformation[1][1] = math.cos(angle)

    new_origin = np.ones((1, 2))
    
    new_origin[0, 0] = -origin[0]*math.cos(angle) + origin[1]*math.sin(angle)
    new_origin[0, 1] = -origin[0]*math.sin(angle) - origin[1]*math.cos(angle)
    origin = np.reshape(origin, (1, 2))
    

    for i in range(len(pixel_coordinates)):
        pixels_array = pixel_coordinates[i]
        transformed_matrix = np.matmul(pixels_array, transformation)
        transformed_matrix += origin + new_origin 
        rotates_pixel_coords.append(np.round(transformed_matrix, 2))
        
    return rotates_pixel_coords
