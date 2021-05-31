import tensorflow as tf
from configuration import IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS, EPOCHS, BATCH_SIZE, save_model_dir, save_every_n_epoch
from prepare_data import generate_datasets, load_and_preprocess_image
from model import MyModel
import math

def process_features(features, data_augmentation):
    image_raw = features['image_raw'].numpy()
    image_tensor_list = []
    for image in image_raw:
        image_tensor = load_and_preprocess_image(image, data_augmentation=data_augmentation)
        image_tensor_list.append(image_tensor)
    # pack the features into a single array
    images = tf.stack(image_tensor_list, axis=0)
    labels = features['label'].numpy()
    
    return images, labels


if __name__ == '__main__':
    
    train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count = generate_datasets()
    
    model = MyModel()
    
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    
    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')
    
    def train_step(image_batch, label_batch):
        with tf.GradientTape() as tape:
            predictions = model(image_batch, training=True)
            loss = loss_object(y_true=label_batch, y_pred = predictions)
            
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))
        
        train_loss.update_state(values=loss)
        train_accuracy.update_state(y_true=label_batch, y_pred=predictions)
        
        return loss
        
    def valid_step(image_batch, label_batch):
        predictions = model(image_batch, training=False)
        v_loss = loss_object(label_batch, predictions)
        
        valid_loss.update_state(values=v_loss)
        valid_accuracy.update_state(y_true=label_batch, y_pred=predictions)
        
    for epoch in range(EPOCHS):
        step=0
        for features in train_dataset:
            step += 1
            images, labels = process_features(features, data_augmentation=True)
            train_step(images, labels)
            print("Epoch: {}/{}, step : {}/{}, loss : {:.3f}, accuracy: {:.3f}".format(epoch,
                                                                                       EPOCHS,
                                                                                       step,
                                                                                       math.ceil(train_count / BATCH_SIZE),
                                                                                       train_loss.result().numpy(),
                                                                                       train_accuracy.result().numpy()))
    
        for features in valid_dataset:
            valid_images, valid_labels = process_features(features, data_augmentation=False)
            valid_step(valid_images, valid_labels)
            
            print("Epoch: {}/{}, train loss: {:.3f}, train accuracy: {:.3f}, valid loss: {:.3f}, valid accuracy: {:.3f}".format(epoch,
                                                                                                                                EPOCHS,
                                                                                                                                train_loss.result().numpy(),
                                                                                                                                train_accuracy.result().numpy(),
                                                                                                                                valid_loss.result().numpy(),
                                                                                                                                valid_accuracy.result().numpy()))
            
        train_loss.reset_states()
        train_accuracy.reset_states()
        valid_loss.reset_states()
        valid_accuracy.reset_states()
        
        if epoch % save_every_n_epoch == 0:
            model.save_weights(filepath=save_model_dir+"epoch-{}".format(epoch),save_format='tf')
            
    model.save_weights(filepath=save_model_dir+"model", save_format='tf')
