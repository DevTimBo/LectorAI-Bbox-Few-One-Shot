from cjm_pytorch_utils.core import get_torch_device, set_seed

import random

import os


from PIL import Image
from faker import Faker

import json
from pathlib import Path
from cjm_pil_utils.core import get_img_files
import pandas as pd
from tqdm.auto import tqdm
#from pdf2image import convert_from_path

import sys

import torch
import io
import re
import sys
from pathlib import Path
import warnings
import shutil
   
project_dir = Path('C:/Users/Anwender/Downloads/LectorAI_SOSE24/final_lectorAI_bbox_project')
handwriting_synth_dir = project_dir / "pytorch-handwriting-synthesis-toolkit"

# Add the directory to sys.path
sys.path.append(str(handwriting_synth_dir))

import handwriting_synthesis.data
from handwriting_synthesis import utils
from handwriting_synthesis.sampling import HandwritingSynthesizer


import matplotlib.pyplot as plt

# Initialize Faker
fake = Faker()


seed = 1234
set_seed(seed)
device = get_torch_device()
dtype = torch.float32
device, dtype


# Function to copy files and resize images if necessary
def copy_files(source_path, destination_folder, target_size=(1024, 1024)):
    source = Path(source_path)
    destination_folder = Path(destination_folder)

    # Ensure the destination directory exists
    destination_folder.mkdir(parents=True, exist_ok=True)

    # Define common image file extensions including uppercase extensions
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.PNG', '.JPG', '.JPEG', '.BMP', '.GIF', '.TIFF']

    if source.is_dir():
        # Iterate over all files in the directory
        for file in source.iterdir():
            destination = destination_folder / file.name
            if file.suffix in image_extensions:
                print(f"Processing image file: {file}")
                # If the file is an image, open and resize it
                try:
                    with Image.open(file) as img:
                        img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
                        # Ensure the destination path has the correct file extension
                        destination_with_ext = destination.with_suffix(file.suffix)
                        img_resized.save(destination_with_ext)
                        print(f"Resized and copied {file} to {destination_with_ext}")
                except Exception as e:
                    print(f"Error processing image file {file}: {e}")
            else:
                # If the file is not an image, just copy it
                try:
                    shutil.copy(file, destination)
                    print(f"Copied {file} to {destination}")
                except Exception as e:
                    print(f"Error copying file {file} to {destination}: {e}")
    else:
        # If the source is a single file, process it directly
        destination = destination_folder / source.name
        if source.suffix in image_extensions:
            print(f"Processing image file: {source}")
            try:
                with Image.open(source) as img:
                    img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
                    destination_with_ext = destination.with_suffix(source.suffix)
                    img_resized.save(destination_with_ext)
                    print(f"Resized and copied {source} to {destination_with_ext}")
            except Exception as e:
                print(f"Error processing image file {source}: {e}")
        else:
            try:
                shutil.copy(source, destination)
                print(f"Copied {source} to {destination}")
            except Exception as e:
                print(f"Error copying file {source} to {destination}: {e}")




def scale_points_in_json(source_path, destination_path, target_size=(1024, 1024)):
    # Load the JSON file
    with open(source_path, 'r') as file:
        data = json.load(file)
    
    # Extract the original image dimensions
    original_width = data['imageWidth']
    original_height = data['imageHeight']
    
    # Calculate scale factors
    scale_x = target_size[0] / original_width
    scale_y = target_size[1] / original_height
    
    # Scale the points in the shapes
    for shape in data['shapes']:
        scaled_points = []
        for point in shape['points']:
            scaled_x = point[0] * scale_x
            scaled_y = point[1] * scale_y
            scaled_points.append([scaled_x, scaled_y])
        shape['points'] = scaled_points
    
    # Update image dimensions
    data['imageWidth'] = target_size[0]
    data['imageHeight'] = target_size[1]
    
    # Save the modified JSON to the destination path
    with open(destination_path, 'w') as file:
        json.dump(data, file, indent=2)
    
    print(f"Scaled points saved to {destination_path}")
    return destination_path

# Function to extract the correct file name from the image path
def extract_filename(path):
    return os.path.splitext(os.path.basename(path.replace('\\', '/')))[0]

# Define the function to get image files
def get_img_files(folder):
    # Define common image file extensions, including uppercase
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif', '*.tiff']
    
    # Create a set to avoid duplicate file paths
    img_files = set()
    

    for ext in extensions:
        img_files.update(Path(folder).glob(ext))
    
    # Convert set to list and print image file paths
    img_files = list(img_files)
    print("Image file paths:", img_files)
    print(f"Current working directory in script: {os.getcwd()}")

    return img_files



def create_test_dataset(source_folder, destination_folder, percentage):
    """
    Move a certain percentage of images and their corresponding JSON files
    from the source folder to the destination folder.

    :param source_folder: The source folder containing images and JSON files
    :param destination_folder: The destination folder to move the files to
    :param percentage: The percentage of files to move
    """
    source_folder = Path(source_folder)
    destination_folder = Path(destination_folder)

    # Create the destination folder if it doesn't exist
    destination_folder.mkdir(parents=True, exist_ok=True)

    # Get a list of all image files in the source folder
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.PNG', '.JPG', '.JPEG', '.BMP', '.GIF', '.TIFF']
    image_files = [file for file in source_folder.iterdir() if file.suffix in image_extensions]

    # Calculate the number of files to move
    num_files_to_move = int(len(image_files) * (percentage / 100))
    
    # Randomly select files to move
    files_to_move = random.sample(image_files, num_files_to_move)

    for image_file in files_to_move:
        # Move the image file
        shutil.move(str(image_file), str(destination_folder / image_file.name))

        # Check for corresponding JSON file and move it
        json_file = image_file.with_suffix('.json')
        if json_file.exists():
            shutil.move(str(json_file), str(destination_folder / json_file.name))
        
        print(f"Moved {image_file.name} and its JSON file to {destination_folder}")

   

def create_dataframe(unique_train_folder):
    # Get a list of image files in the dataset
    img_file_paths = get_img_files(unique_train_folder)

    # Convert the folder path to a Path object
    json_folder_path = Path(unique_train_folder)
    
    # Get a list of JSON files in the dataset
    annotation_file_paths = list(json_folder_path.glob('*.json'))

        # Scale points in each JSON file and overwrite it
    for json_path in annotation_file_paths:
        scale_points_in_json(json_path, json_path)
        
    # Create dictionaries that map filenames (without extensions) to file paths
    img_dict = {file.stem: file for file in img_file_paths}
    annotation_dict = {file.stem: file for file in annotation_file_paths}

    # Create DataFrames from the dictionaries
    img_df = pd.DataFrame.from_dict(img_dict, orient='index', columns=['Image File'])
    annotation_df = pd.DataFrame.from_dict(annotation_dict, orient='index', columns=['Annotation File'])

    # Merge the DataFrames on the index which is the stem of the filenames
    merged_df = pd.merge(img_df, annotation_df, left_index=True, right_index=True, how='inner')
    annotation_df = merged_df
    # Display the merged DataFrame to check the pairing
    # Create a dictionary that maps file names to file paths
    img_dict = {file.stem : file for file in img_file_paths}

    # Display the first five entries from the dictionary using a Pandas DataFrame
    pd.DataFrame.from_dict(img_dict, orient='index')
    # Create a generator that yields Pandas DataFrames containing the data from each JSON file
    cls_dataframes = (pd.read_json(f, orient='index').transpose() for f in tqdm(annotation_file_paths))

    # Concatenate the DataFrames into a single DataFrame
    annotation_df = pd.concat(cls_dataframes, ignore_index=False)

    # Extract the file name without the 'Bilder' folder paths to use as index
    annotation_df['index'] = annotation_df['imagePath'].apply(extract_filename)

    # Set the new index
    annotation_df = annotation_df.set_index('index')
        # Check for discrepancies between the keys of img_dict and the index of annotation_df
    discrepancies = [(key, key in annotation_df.index) for key in img_dict.keys()]
    print(discrepancies)
    # Keep only the rows that correspond to the filenames in the 'img_dict' dictionary
    annotation_df = annotation_df.loc[list(img_dict.keys())]
    shapes_df = annotation_df['shapes'].explode().to_frame().shapes.apply(pd.Series)

    return shapes_df,img_dict,annotation_df

def generate_handwritten_text(model_path, text, bias=3, trials=1, show_weights=False, heatmap=False, thickness=10, output_file_type="png"):
    device = torch.device("cpu")
    synthesizer = HandwritingSynthesizer.load(model_path, device, bias)
    generated_images = []

    base_file_name = re.sub('[^0-9a-zA-Z]+', '_', text)

    if heatmap:
        full_text = text + '\n'
        c = handwriting_synthesis.data.transcriptions_to_tensor(synthesizer.tokenizer, [full_text])
        buffer = io.BytesIO()
        utils.plot_mixture_densities(synthesizer.model, synthesizer.mu, synthesizer.sd, buffer, c)
        buffer.seek(0)
        img = Image.open(buffer)
        generated_images.append(img)
    else:
        for i in range(1, trials + 1):
            buffer = io.BytesIO()
            if show_weights:
                synthesizer.visualize_attention(text, buffer, thickness=thickness)
            else:
                synthesizer.generate_handwriting(text, buffer, thickness=thickness)
            
            buffer.seek(0)
            
            try:
                img = Image.open(buffer)
                img.format = 'PNG'  
                generated_images.append(img)
                print(f'Done {i} / {trials}')
            except Exception as e:
                warnings.warn(f"Error opening image from buffer: {e}")
                continue

    return generated_images[0]

# Function to resize image to 1024x1024
def resize_image(image, target_size=(1024, 1024)):
    return image.resize(target_size, Image.Resampling.LANCZOS)

class Args:
    def __init__(self, model_path, text, bias=2, trials=1, show_weights=False, heatmap=False):
        self.model_path = model_path
        self.text = text
        self.bias = bias
        self.trials = trials
        self.show_weights = show_weights
        self.heatmap = heatmap


args = Args(
    model_path= handwriting_synth_dir / 'checkpoints' / 'Epoch_46', #epoch52,56,46
    text=fake.text(),
    bias=2.0,
    trials=1,
    show_weights=False,
    heatmap=False
)

# Function to create synthetic document
def create_synthetic_document(template_path, output_path, shapes_df, start_range, num_documents):
    font_size = 25
    place_image_size = (200, font_size)
    target_size = (1024, 1024)  
    
    # Load the template image and resize it to 1024x1024
    template = Image.open(template_path).convert('RGBA')
    template_resized = resize_image(template, target_size)
    # Extract base name of the template image
    base_name = os.path.splitext(os.path.basename(template_path))[0]
    
    # Fetch class names from your DataFrame or list
    class_names = shapes_df['label'].unique().tolist()
   
    # Generate synthetic documents
    for doc_num in range(num_documents):
        # Create a copy of the resized template for each document
        document = template_resized.copy()
        
        # Iterate over each class name and fetch corresponding positions
        for class_name in class_names:

            # Fetch positions from shapes_df based on class_name
            points = shapes_df.loc[shapes_df['label'] == class_name, 'points'].iloc[0]
            # Extract the first point for positioning text
            x, y = int(points[0][0]), int(points[0][1])
            
            # Adjust x position differently for the specific label
            if class_name == 'student_akt_sems':
                x_adjusted = x + 300  # Adjust x position for 'student_akt_sems'
            elif class_name == 'student_ECTS':
                x_adjusted = x + 340 
            else:
                x_adjusted = x + 150  # Default x adjustment for other labels
            
            y_adjusted = y  # Adjust y position
            
            # Generate handwritten text image
            text_image = generate_handwritten_text(
                args.model_path,
                args.text,
                bias=args.bias,
                trials=args.trials,
                show_weights=args.show_weights,
                heatmap=args.heatmap
            )
            
            # Resize text_image to the specified size
            text_image = text_image.resize(place_image_size, Image.Resampling.LANCZOS)
                        # Check if text_image is created correctly

            #view_created_sample(text_image)
            # Overlay handwritten text image on the document at the specified position
            document.paste(text_image, (x_adjusted, y_adjusted), text_image)
        
        # Save or display the generated document
        output_file = os.path.join(output_path, f"{base_name}_{doc_num + start_range}.png")
        document.save(output_file)
        print(f"Generated document {doc_num + 1}: {output_file}")


def view_created_sample(sample_image):
    # If sample_image is a path, open it, otherwise assume it's an Image object
    if isinstance(sample_image, str):
        sample_image = Image.open(sample_image).convert('RGBA')
    elif isinstance(sample_image, Image.Image):
        sample_image = sample_image.convert('RGBA')
    else:
        raise ValueError("Invalid input to view_created_sample: expected file path or PIL Image")

    sample_image.show() 
    

def read_json_file(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    return data


def get_image_files_in_folder(folder_path):
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]
    return images


def create_json_copies(json_data, output_folder, image_files):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for image_file in image_files:
        # Remove the image file extension to use as JSON file name
        json_filename = os.path.splitext(image_file)[0] + '.json'
        json_copy_path = os.path.join(output_folder, json_filename)
        
        # Update the JSON data with new image filename
        json_data['imagePath'] = image_file
        
        # Write the modified JSON data to the new JSON file
        with open(json_copy_path, 'w') as file:
            json.dump(json_data, file, indent=4)
        
        print(f"Created copy: {json_copy_path}")


def filter_labels(input_dir, output_dir, labels_to_exclude):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate over all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            # Read the JSON file
            with open(input_path, 'r') as file:
                data = json.load(file)
                
            # Filter the shapes to exclude unwanted labels
            filtered_shapes = [shape for shape in data['shapes'] if shape['label'] not in labels_to_exclude]
            data['shapes'] = filtered_shapes
            
            # Write the modified JSON to the output directory with the same filename
            with open(output_path, 'w') as file:
                json.dump(data, file, indent=4)
                
            print(f"Processed {filename}")
                    

def main(json_path,  output_folder, image_folder_path):
    create_json_copies(read_json_file(json_path), output_folder, get_image_files_in_folder(image_folder_path))
    
    









