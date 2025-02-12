# Load and evaluate .h5 or .tflite model on the test set
import tensorflow as tf
from tqdm import tqdm

SEED = 123

model_path = "quant_aaaabh.tflite"
img_size = (50, 50)
batch_size = 1
test_dir = "/path/to/test" # to be modified

# Load test dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels="inferred",
    label_mode="binary",
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False,
    seed=SEED
)
test_ds = test_ds.map(lambda x, y: (x, y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

#Test quantized model
interpreter = tf.lite.Interpreter(model_path)
interpreter.allocate_tensors()

output = interpreter.get_output_details()[0]  # Model has single output.
input = interpreter.get_input_details()[0]  # Model has single input.

correct = 0
wrong = 0

for image, label in tqdm(test_ds):
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