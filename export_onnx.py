import torch
import torch.onnx
import onnx
import onnxruntime
import numpy as np
from train_mlp_torch import BisindoMLP, INPUT_SIZE, HIDDEN_SIZES, NUM_CLASSES, MODEL_SAVE_PATH

# Configuration
ONNX_MODEL_PATH = "bisindo_mlp.onnx"

def export_model():
    # Load model
    model = BisindoMLP(INPUT_SIZE, HIDDEN_SIZES, NUM_CLASSES)
    try:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file '{MODEL_SAVE_PATH}' not found. Make sure training is complete.")
        return

    model.eval()

    # Create dummy input: 1 batch, 126 features
    dummy_input = torch.randn(1, INPUT_SIZE, requires_grad=True)

    # Export to ONNX
    torch.onnx.export(model,               # model being run
                      dummy_input,         # model input (or a tuple for multiple inputs)
                      ONNX_MODEL_PATH,     # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=12,    # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})

    print(f"Model exported to {ONNX_MODEL_PATH}")

    # Verify ONNX model
    onnx_model = onnx.load(ONNX_MODEL_PATH)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verified.")

    # Test inference with ONNX Runtime
    ort_session = onnxruntime.InferenceSession(ONNX_MODEL_PATH)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    with torch.no_grad():
        torch_out = model(dummy_input)
    
    # compare the results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

if __name__ == "__main__":
    export_model()
