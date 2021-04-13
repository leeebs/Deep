import tensorflow as tf
import argparse
from dataset import Dataset
from model import MyModel, VggModel, MiniInception, ResNet
from loss import loss_fn

class Train(object):
    def __init__(self):
        self.save_path = args.save_model_dir
        self.train_ds = Dataset().split_ds_train()
        self.model = MyModel()
        self.optimizer = tf.keras.optimizers.Adam(args.learning_rate)
        self.train_loss = tf.keras.metrics.Mean()
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        self.train_loss_results = []
        self.train_accuracy_results = []
        self.epochs = args.epochs
        
    @tf.function
    def train_step(self, img, label):
        with tf.GradientTape() as tape:
            logits = self.model(img, training=True)
            loss = loss_fn(label, logits)
            
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        self.train_loss(loss)
        self.train_accuracy(label, logits)
        
        return loss

    def main(self):
        
        self.train_loss.reset_states()
        self.train_accuracy.reset_states()
                
        for epoch in range(self.epochs):
            
            for img, label in self.train_ds:
                self.train_step(img, label)
                
            self.train_loss_results.append(self.train_loss.result())
            self.train_accuracy_results.append(self.train_accuracy.result())
            
            if epoch % 1 == 0:
                print("Epoch {:03d} : Loss {:.3f} , Accuracy {:.3f}".format(epoch,
                                                                            self.train_loss.result(),
                                                                            self.train_accuracy.result()))
                self.model.save_weights(filepath=self.save_path + 'epoch--{}'.format(epoch), save_format="tf")
                
        tf.saved_model.save(self.model, self.save_path)
    
        #print(self.model.summary())

        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_model_dir", type=str, default="ckpt\\")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    args = parser.parse_args()
    
    Train().main()

    
                
            
