from tensorflow_model_optimization.python.core.keras.compat import keras #for Quantization Aware Training (QAT)
import tensorflow_model_optimization as tfmot #for Post Training Quantization (PTQ)
import random
import numpy as np
import tensorflow as tf

# GPU memory allocation
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=3500)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print(e)

#some hyperparameters
img_size = (50, 50)
batch_size = 64
learning_rate = 0.0001
epochs = 100
model_name = "aaaabh_quality"
model_path = "model_aaaabh.h5"

# Define paths to your dataset
train_dir = "wake_vision/train_quality"
validation_dir = "wake_vision/validation"

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Load datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels="inferred",
    label_mode="binary",
    image_size=img_size,
    batch_size=batch_size,
    shuffle=True,
    seed=42,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    labels="inferred",
    label_mode="binary",
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False,
    seed=42,
)

data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomContrast(0.2),
        ])

train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (x, y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

#set validation based early stopping
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath= model_name + ".h5",
    monitor='val_accuracy',
    mode='max', save_best_only=True)


early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=20,
    mode='max',
    restore_best_weights=True
)

# Load the model found by uNAS (https://github.com/Elios-Lab/uNAS)
with tfmot.quantization.keras.quantize_scope():
    model = tf.keras.models.load_model(model_path)
    config = model.get_config()

    qa_model = keras.Model.from_config(config)
    qa_model.summary()
    qa_model.set_weights(model.get_weights())


# Compile the model
qa_model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

try:
    # Train the model
    history = qa_model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=[model_checkpoint_callback, early_stopping_callback]
    )
except KeyboardInterrupt:
    print("\n\nTraining interrupted\n\n")
    qa_model.save(model_name + ".h5")
    
print("\n\nTraining completed\n\n")

print("\n\nExecuting post-training quantization...\n\n")

qmodel_name = "quant_aaaabh"

def representative_dataset():
    for data in train_ds.rebatch(1).take(150) :
        yield [tf.dtypes.cast(data[0], tf.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model = converter.convert()

with open(qmodel_name + ".tflite", 'wb') as f:
    f.write(tflite_model)