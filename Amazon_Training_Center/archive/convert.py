import onnx
from onnx import helper, numpy_helper, TensorProto
import numpy as np
from onnx2torch import convert
import onnxruntime as ort
import torch
import cv2

def preprocess_onnx_model(onnx_model_path, modified_onnx_model_path):
    # Load the ONNX model
    onnx_model = onnx.load(onnx_model_path)
    graph = onnx_model.graph

    # Edit the model to replace GreaterOrEqual with Less and Not operations, and ensure Clip nodes have static min/max values
    new_nodes = []
    for node in graph.node:
        if node.op_type == "GreaterOrEqual":
            input_a = node.input[0]
            input_b = node.input[1]
            output = node.output[0]

            # Create intermediate nodes for replacement
            less_output = output + "_less"

            less_node = helper.make_node(
                "Less",
                inputs=[input_b, input_a],
                outputs=[less_output]
            )

            not_node = helper.make_node(
                "Not",
                inputs=[less_output],
                outputs=[output]
            )

            # Append the new nodes to the list
            new_nodes.extend([less_node, not_node])
        
        elif node.op_type == "Clip":
            min_val = None
            max_val = None
            if len(node.input) >= 2:
                try:
                    min_tensor = [tensor for tensor in graph.initializer if tensor.name == node.input[1]]
                    if min_tensor:
                        min_val = numpy_helper.to_array(min_tensor[0])
                    else:
                        min_val = 0.0  # Default min value if not found
                except:
                    min_val = 0.0  # Default min value if error
            if len(node.input) == 3:
                try:
                    max_tensor = [tensor for tensor in graph.initializer if tensor.name == node.input[2]]
                    if max_tensor:
                        max_val = numpy_helper.to_array(max_tensor[0])
                    else:
                        max_val = 1.0  # Default max value if not found
                except:
                    max_val = 1.0  # Default max value if error

            # Update the Clip node with static values
            if min_val is not None:
                new_min = helper.make_tensor("min", TensorProto.FLOAT, [], [float(min_val)])
                graph.initializer.append(new_min)
                node.input[1] = "min"
            if max_val is not None:
                new_max = helper.make_tensor("max", TensorProto.FLOAT, [], [float(max_val)])
                graph.initializer.append(new_max)
                node.input[2] = "max"
            
            new_nodes.append(node)
        
        elif node.op_type == "If":
            # Handle If node, replace it with a simpler node or static value if possible
            # This is a placeholder to handle specific cases manually
            # Replace the If node with its true or false subgraph, depending on a static condition
            # Assuming a condition that always results in true for simplicity
            true_subgraph = node.attribute[0].g
            new_nodes.extend(true_subgraph.node)

        else:
            new_nodes.append(node)

    # Replace the nodes in the graph with the new nodes
    del graph.node[:]
    graph.node.extend(new_nodes)

    # Save the modified ONNX model
    onnx.save(onnx_model, modified_onnx_model_path)

def convert_onnx_to_torch(modified_onnx_model_path):
    # Load the modified ONNX model
    onnx_model = onnx.load(modified_onnx_model_path)

    # Convert the ONNX model to a PyTorch model
    torch_model = convert(onnx_model)
    return torch_model

def run_inference_and_compare(torch_model, modified_onnx_model_path, image_path):
    # Load input image
    x = cv2.imread(image_path)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = np.transpose(x, (2, 0, 1))  # HWC to CHW
    x = np.expand_dims(x, axis=0)  # Add batch dimension
    x = torch.tensor(x, dtype=torch.float32)

    # Run inference with PyTorch model
    with torch.no_grad():
        out_torch = torch_model(x)

    # Run inference with ONNX Runtime
    ort_sess = ort.InferenceSession(modified_onnx_model_path)
    outputs_ort = ort_sess.run(None, {"input": x.numpy()})

    # Check the Onnx output against PyTorch
    outputs_ort_np = np.array(outputs_ort).squeeze()
    out_torch_np = out_torch.detach().numpy().squeeze()
    print("Max difference:", np.max(np.abs(outputs_ort_np - out_torch_np)))
    print("All close:", np.allclose(outputs_ort_np, out_torch_np, atol=1.0e-7))

def save_torch_model(torch_model, save_path):
    torch.save(torch_model, save_path)

# Paths
onnx_model_path = "maskrcnn.onnx"
modified_onnx_model_path = "/mnt/c/Users/timBo/Desktop/Projects/LectorAI-Bbox-Transferlearning/Amazon_Training_Center/modified_model.onnx"
image_path = "/mnt/c/Users/timBo/Desktop/Projects/LectorAI-Bbox-Transferlearning/Amazon_Training_Center/tf_images/AL_001.jpg"
torch_model_save_path = "/mnt/c/Users/timBo/Desktop/Projects/LectorAI-Bbox-Transferlearning/models/pre_mask_rcnn_weights.pt"

# Execute steps
preprocess_onnx_model(onnx_model_path, modified_onnx_model_path)
torch_model = convert_onnx_to_torch(modified_onnx_model_path)
run_inference_and_compare(torch_model, modified_onnx_model_path, image_path)
save_torch_model(torch_model, torch_model_save_path)
