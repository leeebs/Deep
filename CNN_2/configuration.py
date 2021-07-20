EPOCHS = 10
BATCH_SIZE = 100
NUM_CLASSES = 5
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
CHANNELS = 3

save_model_dir = "tfrecord\\saved_model\\"
save_every_n_epoch = 10
test_image_dir = ""

dataset_dir = "tfrecord\\dataset\\"
train_dir = dataset_dir + "train"
valid_dir = dataset_dir + "valid" 
test_dir = dataset_dir + "test" 
train_tfrecord = dataset_dir + "train.tfrecord"
valid_tfrecord = dataset_dir + "valid.tfrecord"
test_tfrecord = dataset_dir + "test.tfrecord" 

TRAIN_SET_RATIO = 0.6
TEST_SET_RATIO = 0.2
