# gen_ecg_image_from_data_refactored.py
# Place this file inside your ecg-image-kit directory

import os, sys, argparse, json
import random
import csv # Keep import if CSV-related helper functions might be used elsewhere
import qrcode
from PIL import Image
import numpy as np
from scipy.stats import bernoulli
from helper_functions import find_files, read_config_file
from extract_leads import get_paper_ecg
from HandwrittenText.generate import get_handwritten
from CreasesWrinkles.creases import get_creased
from ImageAugmentation.augment import get_augment
import warnings

# Suppress TensorFlow and general warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
warnings.filterwarnings("ignore")

# No longer called by generate_single_ecg_image.
# If these CSVs need to be initialized, handle it once per output directory
# in the main parallel script before starting the pool.
#
# def writeCSV(output_directory, start_index):
#     csv_file_path = os.path.join(output_directory,'Coordinates.csv')
#     if not os.path.isfile(csv_file_path):
#         with open (csv_file_path,'a') as ground_truth_file:
#                 writer = csv.writer(ground_truth_file)
#                 if start_index != -1:
#                     writer.writerow(["Filename","class","x_center","y_center","width","height"])

#     grid_file_path = os.path.join(output_directory,'gridsizes.csv')
#     if not os.path.isfile(grid_file_path):
#         with open (grid_file_path,'a') as gridsize_file:
#             writer = csv.writer(gridsize_file)
#             if start_index != -1:
#                 writer.writerow(["filename","xgrid","ygrid","lead_name","start","end"])


def generate_single_ecg_image(**kwargs):
    """
    Core logic to generate a single ECG image.
    This function accepts all original command-line arguments as keyword arguments.
    It performs the image generation and saves files to the specified output directory.
    
    This function is designed to be called by a multiprocessing worker.
    """
    # Create an 'args' like object from the kwargs dictionary for compatibility
    # with the original code structure that used 'args.argument_name'.
    class Args:
        def __init__(self, **entries):
            self.__dict__.update(entries)
    
    args = Args(**kwargs)

    # The original script had:
    # if hasattr(args, 'st') == True:
    #     random.seed(args.seed)
    #     args.encoding = args.input_file
    #
    # Simplified logic:
    if args.seed != -1: # Only seed if a specific seed is provided
        random.seed(args.seed)
    args.encoding = args.input_file # This is used for QR code data

    filename = args.input_file
    header = args.header_file
    
    # Random parameters (if not deterministic)
    resolution = random.choice(range(50,args.resolution+1)) if (args.random_resolution) else args.resolution
    padding = random.choice(range(0,args.pad_inches+1)) if (args.random_padding) else args.pad_inches
    
    papersize = '' # Not used in original code from context
    lead = args.remove_lead_names # This is True/False directly from argparse

    # Bernoulli distributions for random features
    bernoulli_dc = bernoulli(args.calibration_pulse)
    bernoulli_bw = bernoulli(args.random_bw)
    bernoulli_grid = bernoulli(args.random_grid_present)
    
    # Conditional print header probability
    if args.print_header: # If --print_header is explicitly true, always add
        bernoulli_add_print = bernoulli(1)
    else: # Otherwise, use the random_print_header probability
        bernoulli_add_print = bernoulli(args.random_print_header)
    
    # Font selection
    font_dir = 'Fonts'
    if not os.path.isdir(font_dir) or not os.listdir(font_dir):
        raise FileNotFoundError(f"'{font_dir}' directory not found or is empty. Make sure it's in the correct path relative to the script.")
    font = os.path.join(font_dir, random.choice(os.listdir(font_dir)))
    
    # Grid color logic
    if args.random_bw == 0: # If not black and white
        if not args.random_grid_color: # If not random grid color
            standard_colours = args.standard_grid_color
        else:
            standard_colours = -1 # Indicates random color selection internally
    else: # If black and white, standard_colours should likely be ignored or handled by the underlying functions
        standard_colours = False # Or a specific BW color code if supported by internals

    # Read config file (assumes config.yaml is in the same directory as this script)
    configs = read_config_file(os.path.join(os.getcwd(), args.config_file))

    # --- Call get_paper_ecg (main image generation without distortions) ---
    out_array = get_paper_ecg(
        input_file=filename, header_file=header, configs=configs, 
        mask_unplotted_samples=args.mask_unplotted_samples, start_index=args.start_index, 
        store_configs=args.store_config, store_text_bbox=args.lead_name_bbox, 
        output_directory=args.output_directory, resolution=resolution, papersize=papersize, 
        add_lead_names=lead, add_dc_pulse=bernoulli_dc, add_bw=bernoulli_bw, 
        show_grid=bernoulli_grid, add_print=bernoulli_add_print, pad_inches=padding, 
        font_type=font, standard_colours=standard_colours, full_mode=args.full_mode, 
        bbox=args.lead_bbox, columns=args.num_columns, seed=args.seed,
        shuffle_leads=args.shuffle_leads
    )
    
    for out in out_array: # out is path to the generated image (e.g., 'output_dir/00001_lr-0.png')
        if args.store_config:
            rec_tail, extn = os.path.splitext(out)
            # Ensure the JSON file exists before trying to read it
            json_file_path = rec_tail + '.json'
            if os.path.exists(json_file_path):
                with open(json_file_path, 'r') as file:
                    json_dict = json.load(file)
            else:
                json_dict = {} # Initialize empty if not found, or handle error
        else:
            json_dict = {} # Always initialize, even if not storing

        # Determine if distortions should be applied (based on --fully_random or explicit flags)
        if args.fully_random:
            hw_text = random.choice((True,False))
            wrinkles = random.choice((True,False))
            augment = random.choice((True,False))
        else:
            hw_text = args.hw_text
            wrinkles = args.wrinkles
            augment = args.augment
        
        # --- Handwritten text addition ---
        if hw_text:
            num_words = args.num_words if args.deterministic_num_words else random.choice(range(2,args.num_words+1))
            x_offset = args.x_offset if args.deterministic_offset else random.choice(range(1,args.x_offset+1))
            y_offset = args.y_offset if args.deterministic_offset else random.choice(range(1,args.y_offset+1))

            out = get_handwritten(link=args.link, num_words=num_words, input_file=out, 
                                  output_dir=args.output_directory, x_offset=x_offset, 
                                  y_offset=y_offset, handwriting_size_factor=args.handwriting_size_factor, 
                                  bbox=args.lead_bbox)
        else:
            num_words = 0
            x_offset = 0
            y_offset = 0

        if args.store_config == 2: # Detailed config
            json_dict['handwritten_text'] = bool(hw_text)
            json_dict['num_words'] = num_words
            json_dict['x_offset_for_handwritten_text'] = x_offset
            json_dict['y_offset_for_handwritten_text'] = y_offset
        
        # --- Wrinkles and Creases ---
        if wrinkles:
            ifWrinkles = True
            ifCreases = True # Assuming both are true if --wrinkles is on
            crease_angle = args.crease_angle if args.deterministic_angle else random.choice(range(0,args.crease_angle+1))
            num_creases_vertically = args.num_creases_vertically if args.deterministic_vertical else random.choice(range(1,args.num_creases_vertically+1))
            num_creases_horizontally = args.num_creases_horizontally if args.deterministic_horizontal else random.choice(range(1,args.num_creases_horizontally+1))
            out = get_creased(out, output_directory=args.output_directory, ifWrinkles=ifWrinkles, 
                              ifCreases=ifCreases, crease_angle=crease_angle, 
                              num_creases_vertically=num_creases_vertically, 
                              num_creases_horizontally=num_creases_horizontally, 
                              bbox=args.lead_bbox)
        else:
            crease_angle = 0
            num_creases_horizontally = 0
            num_creases_vertically = 0

        if args.store_config == 2: # Detailed config
            json_dict['wrinkles'] = bool(wrinkles)
            json_dict['crease_angle'] = crease_angle
            json_dict['number_of_creases_horizontally'] = num_creases_horizontally
            json_dict['number_of_creases_vertically'] = num_creases_vertically

        # --- Augmentations (Noise, Rotation, Crop, Temperature) ---
        if augment:
            noise = args.noise if args.deterministic_noise else random.choice(range(1,args.noise+1))
            
            # Crop logic
            if not args.lead_bbox: # If bounding boxes are not needed, allow random crop
                do_crop = bernoulli(args.crop).rvs() # Use bernoulli for True/False based on probability
                if do_crop:
                    crop = args.crop
                else:
                    crop = 0 # If do_crop is False, then crop is 0
            else: # If bounding boxes are needed, crop should be 0
                crop = 0

            # Temperature logic
            blue_temp = random.choice((True,False)) # Randomly choose between two temp ranges
            if blue_temp:
                temp = random.choice(range(2000,4000))
            else:
                temp = random.choice(range(10000,20000))
            
            # Rotation logic
            rotate = args.rotate if args.deterministic_rot else random.choice(range(0, args.rotate + 1)) # Random if not deterministic

            out = get_augment(out, output_directory=args.output_directory, rotate=rotate, 
                              noise=noise, crop=crop, temperature=temp, bbox=args.lead_bbox, 
                              store_text_bounding_box=args.lead_name_bbox, json_dict=json_dict)
        else:
            crop = 0
            temp = 0
            rotate = 0
            noise = 0
        
        if args.store_config == 2: # Detailed config
            json_dict['augment'] = bool(augment)
            json_dict['crop'] = crop
            json_dict['temperature'] = temp
            json_dict['rotate'] = rotate
            json_dict['noise'] = noise

        # --- Store Config (JSON) ---
        if args.store_config:
            json_object = json.dumps(json_dict, indent=4)
            # Use rec_tail from get_paper_ecg to ensure consistent naming
            with open(rec_tail + '.json', "w") as f:
                f.write(json_object)

        # --- Add QR Code ---
        if args.add_qr_code:
            img = np.array(Image.open(out))
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=5,
                border=4,
            )
            qr.add_data(args.encoding)
            qr.make(fit=True)

            qr_img = np.array(qr.make_image(fill_color="black", back_color="white"))
            qr_img_color = np.zeros((qr_img.shape[0], qr_img.shape[1], 3))
            qr_img_color[:,:,0] = qr_img*255.
            qr_img_color[:,:,1] = qr_img*255.
            qr_img_color[:,:,2] = qr_img*255.
            
            # Place QR code in top-right corner
            img[:qr_img.shape[0], -qr_img.shape[1]:, :3] = qr_img_color
            img = Image.fromarray(img)
            img.save(out)

    # Return success status, or number of images generated, as per original
    return "Success" # Original returned len(out_array)

def get_parser():
    """
    Returns an argparse parser for command-line arguments.
    This is kept for standalone script execution.
    """
    parser = argparse.ArgumentParser(description="Generate a single ECG image from a WFDB record with various distortions.")
    parser.add_argument('-i', '--input_file', type=str, required=True, help='Path to the input ECG data file (.dat)')
    parser.add_argument('-hea', '--header_file', type=str, required=True, help='Path to the input ECG header file (.hea)')
    parser.add_argument('-o', '--output_directory', type=str, required=True, help='Path to the output directory to store the synthetic ECG images')
    parser.add_argument('-se', '--seed', type=int, default=-1, help='Seed controlling all the random parameters; default: -1 (no fixed seed)')
    parser.add_argument('-st', '--start_index', type=int, required=False, default=-1, help='Start index for generating image from a single ECG record; default: -1')
    parser.add_argument('--num_leads', type=str, default='twelve', help='Number of leads to plot (e.g., "twelve"); default: "twelve"')
    parser.add_argument('--config_file', type=str, default='config.yaml', help='Path to the configuration YAML file; default: config.yaml')

    # Distortionless image parameters
    parser.add_argument('-r', '--resolution', type=int, default=200, help='Resolution of the generated image in DPI; default: 200')
    parser.add_argument('--pad_inches', type=int, default=0, help='Padding of white border along the image in inches; default: 0')
    parser.add_argument('-ph', '--print_header', action="store_true", default=False, help='Add text from header file on all the generated images; default: False')
    parser.add_argument('--num_columns', type=int, default=-1, help='Number of columns of the ECG leads. -1 for default layout; default: -1')
    parser.add_argument('--full_mode', type=str, default='II', help='Sets the lead to add at the bottom as a long strip; default: "II"')
    parser.add_argument('--mask_unplotted_samples', action="store_true", default=False, help='Mask unplotted samples with NaN in generated WFDB signal file; default: False')
    parser.add_argument('--add_qr_code', action="store_true", default=False, help='Add QR code to all the generated images; default: False')

    # Handwritten text distortions
    parser.add_argument('-l', '--link', type=str, default='', help='URL to capture relevant ECG-related text for handwritten artifacts; default: empty string')
    parser.add_argument('-n', '--num_words', type=int, default=5, help='Number of handwritten words to add; default: 5')
    parser.add_argument('--x_offset', type=int, default=30, help='Horizontal offset (in pixels) of placed words from image border; default: 30')
    parser.add_argument('--y_offset', type=int, default=30, help='Vertical offset (in pixels) of word placement from image top border; default: 30')
    parser.add_argument('--hws', dest='handwriting_size_factor', type=float, default=0.2, help='Handwriting size factor; default: 0.2')
    
    # Wrinkles and creases distortions
    parser.add_argument('-ca', '--crease_angle', type=int, default=90, help='Crease angle (in degrees) with respect to the image; default: 90')
    parser.add_argument('-nv', '--num_creases_vertically', type=int, default=10, help='Number of creases to add vertically; default: 10')
    parser.add_argument('-nh', '--num_creases_horizontally', type=int, default=10, help='Number of creases to add horizontally; default: 10')

    # Augmentation and noise
    parser.add_argument('-rot', '--rotate', type=int, default=0, help='Rotation angle by which images can be rotated; default: 0')
    parser.add_argument('-noise', '--noise', type=int, default=50, help='Noise levels to be added; default: 50')
    parser.add_argument('-c', '--crop', type=float, default=0.01, help='Percentage by which image will be cropped; default: 0.01')
    parser.add_argument('-t', '--temperature', type=int, default=40000, help='Colour temperature changes to be added to the image; default: 40000')
    parser.add_argument('--shuffle_leads', action='store_true', default=False, help='Randomly shuffle the locations of the ECG lead signals on the grid; default: False')

    # Randomization and probabilities
    parser.add_argument('--random_resolution', action="store_true", default=False, help='Generate random resolutions of images; default: False')
    parser.add_argument('--random_padding', action="store_true", default=False, help='Generate random padding widths on images; default: False')
    parser.add_argument('--random_grid_color', action="store_true", default=False, help='Generates random colors for the gridlines; default: False')
    parser.add_argument('--standard_grid_color', type=int, default=5, help='Color of the grid lines (1-5); default: 5 (red)')
    parser.add_argument('--calibration_pulse', type=float, default=1, help='Probability of adding ECG calibration pulse (0-1); default: 1')
    parser.add_argument('--random_grid_present', type=float, default=1, help='Probability of the generated images having the ECG paper grid (0-1); default: 1')
    parser.add_argument('--random_print_header', type=float, default=0, help='Probability of adding printed text to a random set of images (0-1); default: 0')
    parser.add_argument('--random_bw', type=float, default=0, help='Probability to make random set of images black and white (0-1); default: 0')
    parser.add_argument('--remove_lead_names', action="store_false", dest='remove_lead_names', default=True, help='Remove lead names from all generated images; default: False (i.e., keep lead names)')
    
    # Bounding box and config storage
    parser.add_argument('--lead_name_bbox', action="store_true", default=False, help='Store bounding box coordinates for lead names; default: False')
    parser.add_argument('--store_config', type=int, nargs='?', const=1, default=0, help='Store config information for each image in a JSON file (1=high-level, 2=detailed); default: 0')
    parser.add_argument('--lead_bbox', action='store_true', default=False, help='Store bounding box coordinates for every individual ECG lead signal; default: False')

    # Deterministic flags for randomness
    parser.add_argument('--deterministic_offset', action="store_true", default=False, help='Use provided X/Y offset parameters deterministically for handwritten text; default: False')
    parser.add_argument('--deterministic_num_words', action="store_true", default=False, help='Use provided number of words deterministically for handwritten text; default: False')
    parser.add_argument('--deterministic_hw_size', action="store_true", default=False, help='Uses a fixed handwriting size for handwritten text; default: False')
    parser.add_argument('--deterministic_angle', action="store_true", default=False, help='Chooses a fixed crease angle for all images; default: False')
    parser.add_argument('--deterministic_vertical', action="store_true", default=False, help='Adds given number of vertical creases deterministically; default: False')
    parser.add_argument('--deterministic_horizontal', action="store_true", default=False, help='Adds given number of horizontal creases deterministically; default: False')
    parser.add_argument('--deterministic_rot', action="store_true", default=False, help='Adds given amount of rotation to all images deterministically; default: False')
    parser.add_argument('--deterministic_noise', action="store_true", default=False, help='Adds given noise level deterministically to all images; default: False')
    parser.add_argument('--deterministic_crop', action="store_true", default=False, help='Adds given crop level deterministically to all images; default: False')
    parser.add_argument('--deterministic_temp', action="store_true", default=False, help='Adds deterministic temperature level to all images; default: False')

    # Overall distortion flags
    parser.add_argument('--fully_random', action='store_true', default=False, help='Apply all distortions randomly; default: False')
    parser.add_argument('--hw_text', action='store_true', default=False, help='Enable handwritten text distortions; default: False')
    parser.add_argument('--wrinkles', action='store_true', default=False, help='Enable wrinkles and creases distortions; default: False')
    parser.add_argument('--augment', action='store_true', default=False, help='Enable augmentation and noise; default: False')

    return parser

def main():
    """
    Main function for standalone execution of this script.
    Parses arguments and calls the core image generation function.
    Handles 'os.chdir' for standalone mode.
    """
    # This ensures relative paths (e.g., 'Fonts') work correctly when run standalone
    path = os.path.join(os.getcwd(), sys.argv[0])
    parentPath = os.path.dirname(path)
    os.chdir(parentPath)

    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])

    # Convert the argparse Namespace object to a dictionary
    kwargs = vars(args)
    generate_single_ecg_image(**kwargs)

if __name__ == '__main__':
    main()