import os
import tensorflow as tf
from object_detection import model_lib_v2
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

# Set paths for TFRecord files and model
TRAINING_DATA_DIR = 'data'
PIPELINE_CONFIG_PATH = 'models/ssd_mobilenet_v1_coco/pipeline.config'
MODEL_DIR = 'models/ssd_mobilenet_v1_coco'

def main(_):
    # Set configurations
    configs = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(PIPELINE_CONFIG_PATH, 'r') as f:
        proto_str = f.read()
        text_format.Merge(proto_str, configs)

    # Update the configs if necessary
    configs.model.ssd.num_classes = 3  # Update with your number of classes
    configs.train_config.batch_size = 24  # Adjust based on your GPU memory
    configs.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(TRAINING_DATA_DIR, 'train.record')]
    configs.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(TRAINING_DATA_DIR, 'eval.record')]
    configs.eval_config.num_examples = 800  # Adjust based on your evaluation data size

    # Write updated config file
    config_text = text_format.MessageToString(configs)
    with tf.io.gfile.GFile(PIPELINE_CONFIG_PATH, 'wb') as f:
        f.write(config_text)

    # Train the model
    tf.config.set_visible_devices([], 'GPU')  # Uncomment if you want to use CPU for training
    model_lib_v2.train_loop(
        pipeline_config_path=PIPELINE_CONFIG_PATH,
        model_dir=MODEL_DIR,
        train_steps=None,
        checkpoint_every_n=1000,
        record_summaries=True)

if __name__ == '__main__':
    tf.compat.v1.app.run()
