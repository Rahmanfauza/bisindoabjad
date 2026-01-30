import os
import subprocess
import sys

def convert_onnx_to_tflite(onnx_path, output_dir="tflite_model"):
    """
    Converts an ONNX model to TFLite using onnx2tf.
    Requires: pip install tensorflow onnx2tf onnx-simplifier onnx_graphsurgeon
    """
    if not os.path.exists(onnx_path):
        print(f"Error: ONNX model not found at {onnx_path}")
        return

    print(f"Starting conversion of {onnx_path}...")
    
    # Command for onnx2tf
    # -i: input onnx file
    # -o: output directory
    # --non_verbose: reduce output noise
    command = [
        "onnx2tf",
        "-i", onnx_path,
        "-o", output_dir,
        "--non_verbose"
    ]

    try:
        # Run the conversion command
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Successfully converted model. Files are in: {output_dir}")
            
            # Find the .tflite file in the output directory
            tflite_file = None
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    if file.endswith(".tflite"):
                        tflite_file = os.path.join(root, file)
                        break
            
            if tflite_file:
                final_name = "bisindo_mlp.tflite"
                # Move/Rename to root directory for easy access
                os.rename(tflite_file, final_name)
                print(f"Model saved as: {final_name}")
            else:
                print("Warning: Conversion finished but .tflite file was not found in output directory.")
        else:
            print("Error during conversion:")
            print(result.stdout)
            print(result.stderr)
            
    except FileNotFoundError:
        print("\nError: 'onnx2tf' command not found.")
        print("Please install dependencies first:")
        print("pip install tensorflow onnx2tf onnx-simplifier onnx_graphsurgeon")

if __name__ == "__main__":
    ONNX_PATH = "bisindo_mlp.onnx"
    convert_onnx_to_tflite(ONNX_PATH)
