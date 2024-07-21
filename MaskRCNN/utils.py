
import onnxruntime as ort
import onnx
from onnxsim import simplify
# Import Python Standard Library dependencies
import json
import math
from pathlib import Path
from pathlib import Path
import matplotlib.pyplot as plt

# Import utility functions
from cjm_pil_utils.core import resize_img
from cjm_pytorch_utils.core import get_torch_device, move_data_to_device

# Import the distinctipy module

# Import matplotlib for creating plots
import matplotlib.pyplot as plt

# Import numpy

# Import the pandas package
import pandas as pd

# Set options for Pandas DataFrame display
pd.set_option('max_colwidth', None)  # Do not truncate the contents of cells in the DataFrame
pd.set_option('display.max_rows', None)  # Display all rows in the DataFrame
pd.set_option('display.max_columns', None)  # Display all columns in the DataFrame

# Import PIL for image manipulation
from PIL import Image, ImageDraw

# Import PyTorch dependencies
import torch
from torch.amp import autocast
#from torchtnt.utils import get_module_summary
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.tv_tensors import BoundingBoxes, Mask
import torchvision.transforms.v2  as transforms

# Import tqdm for progress bar
from tqdm.auto import tqdm
train_sz = 1024
device = get_torch_device()
dtype = torch.float32
device, dtype

image_size = train_sz
# Set training image size
train_sz = image_size

# Learning rate for the model
lr = 5e-4

# Number of training epochs
epochs = 70

class EarlyStopping:
    def __init__(self, patience, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
                
def plot_label_distribuition(class_counts):
    # Plot the distribution
    plt.figure(figsize=(15, 8))  
    plt.bar(range(len(class_counts)), class_counts.values, align='center')
    plt.title('Class distribution')
    plt.ylabel('Count')
    plt.xlabel('Classes')
    plt.xticks(range(len(class_counts.index)), class_counts.index, rotation=75, ha='right')  # Rotate labels and align right
    plt.tight_layout()  
    plt.show()
    
    
def plot_metrics(train_losses, valid_losses, learning_rates):
    epochs = range(1, len(train_losses) + 1)

    # Plot all metrics on the same figure
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # Training and validation losses
    ax1.plot(epochs, train_losses, label='Training Loss', color='red')
    ax1.plot(epochs, valid_losses, label='Validation Loss', color='blue')
    
    # Secondary y-axis for learning rates
    ax2 = ax1.twinx()
    ax2.plot(epochs, learning_rates, label='Learning Rate', color='green')
    
    # Title and labels
    plt.title(' image_size:1024x1024, batch_size:2, dataset:171, all main and sub boxes AD and AG')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax2.set_ylabel('Learning Rate')
    
    # Legends (adjust position)
    ax1.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0))
    ax2.legend(loc='upper left', bbox_to_anchor=(0.8, 1.0))
    
    # Show plot
    plt.show()
    
def adjust_display_test_image(img_dict_test, index):
      # Convert dictionary keys to a list
    keys = list(img_dict_test.keys())
    
    # Get the key corresponding to the index
    if index < 0 or index >= len(keys):
        print("Index out of range")
        return
    
    file_id = keys[index]
    
    # Retrieve the image file path associated with the file ID
    test_file = img_dict_test[file_id]

    # Open the test file
    test_img = Image.open(test_file).convert('RGB')

    # Resize the test image
    input_img = resize_img(test_img, target_sz=train_sz, divisor=1)

    # Calculate the scale between the source image and the resized image
    min_img_scale = min(test_img.size) / min(input_img.size)

    display(test_img)
    
    return input_img, min_img_scale,test_img    
    
    
def get_test_image_labels(annotation_df, file_id, test_img, input_img, min_img_scale):

    # Print the prediction data as a Pandas DataFrame for easy formatting
    pd.Series({
        "Source Image Size:": input_img.size,
        "Input Dims:": input_img.size,
        "Min Image Scale:": min_img_scale,
        "Input Image Size:": input_img.size
    }).to_frame().style.hide(axis='columns')
        # Extract the polygon points for segmentation mask
    target_shape_points = [shape['points'] for shape in annotation_df.loc[file_id]['shapes']]
    # Format polygon points for PIL
    target_xy_coords = [[tuple(p) for p in points] for points in target_shape_points]
    # Generate mask images from polygons
    target_mask_imgs = [create_polygon_mask(test_img.size, xy) for xy in target_xy_coords]
    # Convert mask images to tensors
    target_masks = Mask(torch.concat([Mask(transforms.PILToTensor()(mask_img), dtype=torch.bool) for mask_img in target_mask_imgs]))

    # Get the target labels and bounding boxes
    target_labels = [shape['label'] for shape in annotation_df.loc[file_id]['shapes']]
    target_bboxes = BoundingBoxes(data=torchvision.ops.masks_to_boxes(target_masks), format='xyxy', canvas_size=test_img.size[::-1])
    return target_labels, target_bboxes, target_masks



def create_polygon_mask(image_size, vertices):
    """
    Create a grayscale image with a white polygonal area on a black background.

    Parameters:
    - image_size (tuple): A tuple representing the dimensions (width, height) of the image.
    - vertices (list): A list of tuples, each containing the x, y coordinates of a vertex
                        of the polygon. Vertices should be in clockwise or counter-clockwise order.

    Returns:
    - PIL.Image.Image: A PIL Image object containing the polygonal mask.
    """

    # Create a new black image with the given dimensions
    mask_img = Image.new('L', image_size, 0)

    # Draw the polygon on the image. The area inside the polygon will be white (255).
    ImageDraw.Draw(mask_img, 'L').polygon(vertices, fill=(255))

    # Return the image with the drawn polygon
    return mask_img



def run_epoch(model, dataloader, optimizer, lr_scheduler, device, scaler, epoch_id, is_training):
    """
    Function to run a single training or evaluation epoch.
    
    Args:
        model: A PyTorch model to train or evaluate.
        dataloader: A PyTorch DataLoader providing the data.
        optimizer: The optimizer to use for training the model.
        loss_func: The loss function used for training.
        device: The device (CPU or GPU) to run the model on.
        scaler: Gradient scaler for mixed-precision training.
        is_training: Boolean flag indicating whether the model is in training or evaluation mode.
    
    Returns:
        The average loss for the epoch.
    """
    # Set the model to training mode
    model.train()
    
    epoch_loss = 0  # Initialize the total loss for this epoch
    progress_bar = tqdm(total=len(dataloader), desc="Train" if is_training else "Eval")  # Initialize a progress bar
    
    # Loop over the data
    for batch_id, (inputs, targets) in enumerate(dataloader):
        # Move inputs and targets to the specified device
        inputs = torch.stack(inputs).to(device)
        
        # Forward pass with Automatic Mixed Precision (AMP) context manager
        with autocast(torch.device(device).type):
            if is_training:
                losses = model(inputs.to(device), move_data_to_device(targets, device))
            else:
                with torch.no_grad():
                    losses = model(inputs.to(device), move_data_to_device(targets, device))
        
            # Compute the loss
            loss = sum([loss for loss in losses.values()])  # Sum up the losses

        # If in training mode, backpropagate the error and update the weights
        if is_training:
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                old_scaler = scaler.get_scale()
                scaler.update()
                new_scaler = scaler.get_scale()
                if new_scaler >= old_scaler:
                    lr_scheduler.step()
            else:
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                
            optimizer.zero_grad()

        # Update the total loss
        loss_item = loss.item()
        epoch_loss += loss_item
        
        # Update the progress bar
        progress_bar_dict = dict(loss=loss_item, avg_loss=epoch_loss/(batch_id+1))
        if is_training:
            progress_bar_dict.update(lr=lr_scheduler.get_last_lr()[0])
        progress_bar.set_postfix(progress_bar_dict)
        progress_bar.update()

        # If loss is NaN or infinity, stop training
        if is_training:
            stop_training_message = f"Loss is NaN or infinite at epoch {epoch_id}, batch {batch_id}. Stopping training."
            assert not math.isnan(loss_item) and math.isfinite(loss_item), stop_training_message

    # Cleanup and close the progress bar 
    progress_bar.close()
    
    # Return the average loss for this epoch
    return epoch_loss / (batch_id + 1)




def train_loop(model, 
               train_dataloader, 
               valid_dataloader, 
               optimizer,  
               lr_scheduler, 
               device, 
               epochs, 
               checkpoint_path, 
               patience, 
               use_scaler=False):
    """
    Main training loop.
    
    Args:
        model: A PyTorch model to train.
        train_dataloader: A PyTorch DataLoader providing the training data.
        valid_dataloader: A PyTorch DataLoader providing the validation data.
        optimizer: The optimizer to use for training the model.
        lr_scheduler: The learning rate scheduler.
        device: The device (CPU or GPU) to run the model on.
        epochs: The number of epochs to train for.
        checkpoint_path: The path where to save the best model checkpoint.
        use_scaler: Whether to scale gradients when using a CUDA device
    
    Returns:
        dict: Dictionary containing lists of training losses, validation losses, and learning rates for each epoch.
    """
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' and use_scaler else None
    best_loss = float('inf')

    early_stopping = EarlyStopping(patience=patience)

    # Initialize lists to store losses and learning rates
    train_losses = []
    valid_losses = []
    learning_rates = []

    for epoch in tqdm(range(epochs), desc="Epochs"):
        train_loss = run_epoch(model, train_dataloader, optimizer, lr_scheduler, device, scaler, epoch, is_training=True)
        with torch.no_grad():
            valid_loss = run_epoch(model, valid_dataloader, None, None, device, scaler, epoch, is_training=False)

        # Append losses and learning rate to lists
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        learning_rates.append(lr_scheduler.get_last_lr()[0])

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)

            training_metadata = {
                'epoch': epoch,
                'train_loss': train_loss,
                'valid_loss': valid_loss, 
                'learning_rate': lr_scheduler.get_last_lr()[0],
                'model_architecture': model.name
            }
            with open(Path(checkpoint_path.parent/'training_metadata.json'), 'w') as f:
                json.dump(training_metadata, f)

               # Check early stopping condition
        early_stopping(valid_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break    
            
    if device.type != 'cpu':
        getattr(torch, device.type).empty_cache()
    
    return {'train_losses': train_losses, 'valid_losses': valid_losses, 'learning_rates': learning_rates}


def export_model_to_onnx(model,onnx_file_path):
   # Set the model to evaluation mode
    model.eval()
    
    input_tensor = torch.randn(1, 3, train_sz, train_sz)
  
    # Export the PyTorch model to ONNX format
    torch.onnx.export(model.cpu(),
                  input_tensor.cpu(),
                  onnx_file_path,
                  export_params=True,
                  do_constant_folding=False,
                  input_names = ['input'],
                  output_names = ['boxes', 'labels', 'scores', 'masks'],
                  dynamic_axes={'input': {2 : 'height', 3 : 'width'}}
                 )
    
    # Move the model to the specified device
    model.to(device)
    # Load the ONNX model from the onnx_file_name
    onnx_model = onnx.load(onnx_file_path)

    # Simplify the model
    model_simp, check = simplify(onnx_model)

    # Save the simplified model to the onnx_file_name
    onnx.save(model_simp, onnx_file_path)



def prediction_on_image_inference(test_img, test_file, onnx_file_path, class_names, checkpoint_dir,threshold=0.8):
   
        # The colormap path
    colormap_path = list(checkpoint_dir.glob('*colormap.json'))[0]

    # Load the JSON colormap data
    with open(colormap_path, 'r') as file:
            colormap_json = json.load(file)

    # Convert the JSON data to a dictionary        
    colormap_dict = {item['label']: item['color'] for item in colormap_json['items']}

    # Extract the class names from the colormap
    class_names = list(colormap_dict.keys())

    # Make a copy of the colormap in integer format
    int_colors = [tuple(int(c*255) for c in color) for color in colormap_dict.values()]
    

    train_sz = 1024

    # Open the test file
    test_img = Image.open(test_file).convert('RGB')

    # Resize the test image
    input_img = resize_img(test_img, target_sz=train_sz, divisor=1)

    # Calculate the scale between the source image and the resized image
    min_img_scale = min(test_img.size) / min(input_img.size)
    
    # Load the model and create an InferenceSession
    session = ort.InferenceSession(onnx_file_path, providers=['CPUExecutionProvider'])

    # Define the transformation to convert the image to a tensor and normalize it
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Apply the transformation to the test image
    input_tensor = transform(test_img).unsqueeze(0).cpu().numpy()


    model_output = session.run(None, {"input": input_tensor})

    # Filter the output based on the confidence threshold
    scores_mask = model_output[2] > threshold
    
    label_list = [class_names[int(idx)] for idx in model_output[1][scores_mask]]

    # Scale the predicted bounding boxes
    pred_bboxes = (model_output[0][scores_mask])*min_img_scale

    # Get the class names for the predicted label indices
    pred_labels = [class_names[int(idx)] for idx in model_output[1][scores_mask]]

    # Extract the confidence scores
    pred_scores = model_output[2]
    
    colors = [int_colors[class_names.index(i)] for i in label_list]

    # Extract the cropped image regions based on the bounding boxes
    cropped_images = []
    for box in pred_bboxes:
        box = box.astype(int).tolist()  # Convert tensor to list of integers
        cropped_image = test_img.crop((box[0], box[1], box[2], box[3]))
        cropped_images.append(cropped_image)

    # Combine the bounding boxes, cropped images, labels, and confidence scores
    results = []
    for label, box, image, score in zip(pred_labels, pred_bboxes, cropped_images, pred_scores):
        results.append((label, box, image, score.item()))

    return results

    