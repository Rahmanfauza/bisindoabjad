import numpy as np
import tensorflow as tf

def verify_tflite(model_path="bisindo_mlp.tflite"):
    if not os.path.exists(model_path):
        print(f"Error: TFLite model '{model_path}' not found.")
        return

    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output details.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"Input Shape: {input_details[0]['shape']}")
    print(f"Output Shape: {output_details[0]['shape']}")

    # Test with dummy data
    input_shape = input_details[0]['shape']
    dummy_input = np.random.random_sample(input_shape).astype(np.float32)
    
    interpreter.set_tensor(input_details[0]['index'], dummy_input)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    print("Inference successful!")
    print(f"Output (first 5 classes): {output_data[0][:5]}")

if __name__ == "__main__":
    import os
    verify_tflite()
