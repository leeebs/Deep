import tensorflow as tf
import os
import pathlib
import numpy as np
import argparse

from tensorflow import keras

class Dataset():
    def __init__(self):
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--img_height", type=int, default=20)
        parser.add_argument("--img_width", type=int, default=20)
        parser.add_argument("--buffer", type=int, default=1000)
        parser.add_argument("--path", type=str, default='flower_photos\\')
        parser.add_argument("--channels", type=int, default=3)
        args = parser.parse_args()
        
        self.dataset_dir = args.path
        self.path = pathlib.Path(args.path)
        self.img_height = args.img_height
        self.img_width = args.img_width
        self.buffer = args.buffer
        self.batch_size = args.batch_size
        self.img_count = len(list(self.path.glob('*/*.jpg')))
        self.channels = args.channels
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE

    def list_dataset(self):
        list_ds = tf.data.Dataset.list_files(str(self.path/'*/*'), shuffle=False)
        list_ds = list_ds.shuffle(self.img_count, reshuffle_each_iteration=False)
        return list_ds

    def get_img_label(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        class_names = np.array(sorted([item.name for item in self.path.glob("*")
                                       if item.name != "LICENSE.txt"]))
        one_hot = parts[-2] == class_names
        label = tf.argmax(one_hot)
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, self.channels)
        img = tf.image.resize(img, [self.img_height, self.img_width])
        img = tf.cast(img, tf.float32) / 255.
        return img, label
    
    #def get_img_and_label(self, file_path):
    
    #def get_label_name(self):
    #    label_name = []
    #    for item in os.listdir(self.dataset_dir):
    #        item_path = os.path.join(self.dataset_dir, item)
    #        if os.path.isdir(item_path):
    #            label_names.append(item)
    #    return label_name
        
    def data_augmentation(self, img, label):
        #img = tf.image.resize_with_crop_or_pad(img, self.img_height + 6, self.img_width + 6)
        img = tf.image.random_crop(
            img, size = [self.img_height, self.img_width, 3])
        img = tf.image.random_brightness(
            img, max_delta = 0.5)
        #img = tf.clip_by_value(img, 0, 1)
        return img, label
    
    def preprocess_ds(self, ds):
        ds = ds.shuffle(self.buffer)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=self.AUTOTUNE)
        return ds
    
    def split_ds_train(self):
        train_ds = self.list_dataset().skip(int(self.img_count * 0.2))
        train_ds = train_ds.map(self.get_img_label, num_parallel_calls=self.AUTOTUNE)
        train_ds = train_ds.map(self.data_augmentation, num_parallel_calls=self.AUTOTUNE)
        train_ds = self.preprocess_ds(train_ds)
        return train_ds
    
    def split_ds_val(self):
        val_ds = self.list_dataset().take(int(self.img_count * 0.2))
        val_ds = val_ds.map(self.get_img_label, num_parallel_calls=self.AUTOTUNE)
        val_ds = self.preprocess_ds(val_ds)
        return val_ds
"""    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_height", type=int, default=20)
    parser.add_argument("--img_width", type=int, default=20)
    parser.add_argument("--buffer", type=int, default=1000)
    parser.add_argument("--path", type=str, default='flower_photos\\')
    parser.add_argument("--channels", type=int, default=3)
    args = parser.parse_args()
"""    
    
    
    
    
    # 함수명 parse 호출 메인 세가지 수정
        #augmentation을 train에만 추가하면 ds을 train, val 한번에 처리?
        #d = Dataset()
        #d = Dataset()
        #d.preprocess_ds
        #d.list_dataset
        #ls = d.split_ds_train()
        #labels = d.get_label()
        #print(f'>>> data(file_path) : {path}')
        #label = Dataset()
        #label.get_label()
"""
        def split_ds(self):
            train_ds = self.list_dataset().skip(int(self.img_count * 0.2))
            val_ds = self.list_dataset().take(int(self.img_count * 0.2))
            train_ds = train_ds.map(self.process_path, num_parallel_calls=self.AUTOTUNE)
            val_ds = val_ds.map(self.process_path, num_parallel_calls=self.AUTOTUNE)
            train_ds = train_ds.map(self.augment, num_parallel_calls=self.AUTOTUNE)
            train_ds = self.configure_for_performance(train_ds)
            val_ds = self.configure_for_performance(val_ds)
            return train_ds, val_ds
"""