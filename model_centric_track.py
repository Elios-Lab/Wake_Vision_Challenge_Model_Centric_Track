from tensorflow_model_optimization.python.core.keras.compat import keras #for Quantization Aware Training (QAT)
import tensorflow_model_optimization as tfmot #for Post Training Quantization (PTQ)
# from datasets import load_dataset #for downloading the Wake Vision Dataset
import tensorflow as tf #for designing and training the model 
from tqdm import tqdm

model_path = "quant_aaaabh.tflite"

#some hyperparameters 
#Play with them!
input_shape = (50,50,3)
batch_size = 1

#load dataset
ds = load_dataset("Harvard-Edge/Wake-Vision")

test_ds = ds['test'].to_tf_dataset(columns='image', label_cols='person')

#some preprocessing 
data_preprocessing = tf.keras.Sequential([
    #resize images to desired input shape
    tf.keras.layers.Resizing(input_shape[0], input_shape[1])])

test_ds = test_ds.map(lambda x, y: (data_preprocessing(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE).batch(1).prefetch(tf.data.AUTOTUNE)

#Test quantized model
interpreter = tf.lite.Interpreter(model_path)
interpreter.allocate_tensors()

output = interpreter.get_output_details()[0]  # Model has single output.
input = interpreter.get_input_details()[0]  # Model has single input.

correct = 0
wrong = 0

for image, label in tqdm(test_ds) :
    # Check if the input type is quantized, then rescale input data to uint8
    if input['dtype'] == tf.uint8:
       input_scale, input_zero_point = input["quantization"]
       image = image / input_scale + input_zero_point
       image = tf.dtypes.cast(image, tf.uint8)
       
    interpreter.set_tensor(input['index'], image)
    interpreter.invoke()
       
    # Modified metric calculation for uint8 linear output
    scaled_output = interpreter.get_tensor(output['index']) / 255.
    predictions = 1 if (scaled_output >= .5) else 0
    if label.numpy() == predictions:
        correct = correct + 1
    else:
        wrong = wrong + 1
            
print(f"\n\nTflite model test accuracy: {correct/(correct+wrong)}\n")