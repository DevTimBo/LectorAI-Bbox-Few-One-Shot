import onnx
from onnx2pytorch import ConvertModel
import torch

def convert_onnx_to_torch(weights_path):
    # Load ONNX model
    onnx_model = onnx.load(f'{weights_path}.onnx')
    
    # Convert ONNX model to PyTorch model
    torch_model = ConvertModel(onnx_model)
    
    # Save PyTorch model
    torch.save(torch_model.state_dict(), f'{weights_path}.pth')
    print(f"Model saved to {weights_path}.pth")

weights_path = 'path' # Change to path to your onnx model
convert_onnx_to_torch(weights_path)
