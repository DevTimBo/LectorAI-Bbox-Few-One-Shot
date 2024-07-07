import onnx
from onnx2pytorch import ConvertModel

# Load the ONNX model
onnx_model = onnx.load("maskrcnn.onnx")

# Convert to PyTorch model
torch_model = ConvertModel(onnx_model)
import torch
torch.save(torch_model.state_dict(), "maskrcnn.pth")