import tensorflow as tf

#read the tfrecord file

def parse_image_function(example_proto):
    # parse the input tf.Example proto
    return tf.io.parse_single_example(example_proto, {
        'label': tf.io.FixedLenFeature([], tf.dtypes.int64),  
        'image_raw': tf.io.FixedLenFeature([], tf.dtypes.string),
    })
      
def get_parsed_dataset(tfrecord_name):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_name)
    parsed_dataset = raw_dataset.map(parse_image_function)
    return parsed_dataset 
    
