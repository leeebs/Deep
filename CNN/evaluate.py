import tensorflow as tf
import argparse

from dataset import Dataset
from loss import loss_fn

class Test():
    def __init__(self):
        
        self.save_path = args.saved_model_dir
        self.val_ds = Dataset().split_ds_val()
        self.model = tf.saved_model.load(self.save_path)
        self.test_loss = tf.keras.metrics.Mean()
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        self.test_loss_results = [] 
        self.test_accuracy_results = []  
        
    @tf.function 
    def test_step(self, img, label):
        logits = self.model(img, training=False)
        loss = loss_fn(label, logits) 
        
        self.test_loss(loss)
        self.test_accuracy(label, logits)
        
        return loss
    
    def main(self):
    
        for epoch in range(1):
            
            for img, label in self.val_ds:
                self.test_step(img, label)
                
            self.test_loss_results.append(self.test_loss.result())
            self.test_accuracy_results.append(self.test_accuracy.result())
            
            print("test >>> Epoch {:03d} : Loss {:.3f} , Accuracy {:.3f}".format(epoch,
                                                                                self.test_loss.result(),
                                                                                self.test_accuracy.result()))
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_model_dir", type=str, default="ckpt\\")
    args = parser.parse_args()
    
    Test().main()
        
