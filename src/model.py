import numpy as np

import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras import layers as tfl
from tensorflow.keras import Model
from tensorflow.keras.applications import Xception, EfficientNetB0, DenseNet121, MobileNetV2 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import sys
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

from utils import thresh_function, random_mean_oversampling, get_data

def get_model(metrics: list,
              checkpoint_path: str,
              backbone_name: str = "EfficientNetB0",
              optim_name: str = "Adam",
              w_init: str = "imagenet",
              trainable_layers: int = 5,
              lr: float = 1e-3,
             ):

    # initialize callbacks

    model_ckpt= ModelCheckpoint(
        checkpoint_path, 
        monitor='val_auc', 
        verbose=0, 
        save_best_only=True,
        mode='max', 
        save_freq='epoch')
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_auc', 
        factor=0.2,                   
        patience=4, 
        min_lr=0.1*lr)
    
    es = EarlyStopping(
        monitor='val_loss', 
        mode='auto', 
        verbose=1,
        restore_best_weights=True, 
        patience=10)
    
    # choose the backbone model
    if backbone_name == "EfficientNetB0":
        backbone = EfficientNetB0(weights=w_init, include_top=False, input_shape=(64,64,3))
    elif backbone_name == "Xception":
        backbone = Xception(weights=w_init, include_top=False, input_shape=(64,64,3))
    elif backbone_name == "MobileNetV2":
        backbone = MobileNetV2(weights=w_init, include_top=False, input_shape=(64,64,3))
    elif backbone_name == "DenseNet121":
        backbone = DenseNet121(weights=w_init, include_top=False, input_shape=(64,64,3))
    else:
        backbone = EfficientNetB0(weights=w_init, include_top=False, input_shape=(64,64,3))
        raise NotImplementedError("This model type is not supported, defaulting to EfficientNet.")
    
    # choose the optimizer
    if optim_name == "Adam":
        optim = Adam(learning_rate=lr)
    elif optim_name == "RMSprop":
        optim = RMSprop(learning_rate=lr)
    elif optim_name == "SGD":
        optim = SGD(learning_rate=lr)
    else:
        optim = Adam(learning_rate=lr)
        raise NotImplementedError("This optimizer type is not supported, defaulting to Adam.")
 
    # set the deepest <trainable_layers> layers in the backbone to the trainable mode
    for n, layer in enumerate(backbone.layers):
        if len(backbone.layers) - n > trainable_layers:
            layer.trainable = False
        else:
            layer.trainable = True
            
    # build the model
    x = backbone.output
    x = tfl.GlobalAveragePooling2D()(x)
    # let's add a dropout layer
    x = tfl.Dropout(0.2)(x)
    # and a logistic layer
    predictions = tfl.Dense(2, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=backbone.input, outputs=predictions)
    callbacks = [model_ckpt, reduce_lr, es]

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(
        optimizer = optim,
        loss ='categorical_crossentropy',
        metrics=metrics)
    
    return model, callbacks

def get_class_weights(x_train, y_train, beta: float = 0.99):
    
    # effective number of samples
    unique_classes = np.unique(y_train)
    N = [len(y_train[y_train == c]) for c in unique_classes]
    
    weights = [(1-beta)/(1-np.power(beta, n_i)) for n_i in N]
    
    return {c:weights[c] for c in unique_classes}
  
def model_training(x_train: np.ndarray,
                   y_train: np.ndarray,
                   paths: list,
                   exp_name: str = "exp_0",
                   rotation_range: float = 20.,
                   mean_ovs: int = 0,
                   backbone_name: str = "EfficientNetB0",
                   optim_name: str = "Adam",
                   w_init: str = 'imagenet',
                   trainable_layers: int = 5,
                   bs: int = 128,
                   lr: float = 1e-3,
                   beta: float = 0.99,
                   use_thresholding: bool = True,):
    
    # if true, use thresholding on the input images
    if use_thresholding:
        p_function = thresh_function
    else:
        p_function = lambda x: x

    metrics = [
      Precision(name='Pr'),
      Recall(name='Re'),
      AUC(name='AUC'),
      AUC(name='PRC', curve='PR')
    ]

    # initialize data generator
    gen = ImageDataGenerator(
        horizontal_flip = True,
        vertical_flip = True,
        rotation_range = rotation_range,
        validation_split = 0.2,
        preprocessing_function = thresh_function,
    )

    # compute class weights
    class_weight = get_class_weights(x_train, y_train)

    # oversample the data if necessary
    if mean_ovs > 0:
        x_train, y_train = random_mean_oversampling(x_train, y_train, mean_ovs)

    y_train = to_categorical(y_train)

    # fit the generator
    gen.fit(x_train)

    # get the model
    model, callbacks = get_model(metrics, 
                                 paths["checkpoint_path"],
                                 backbone_name,
                                 optim_name,
                                 w_init,
                                 trainable_layers,
                                 lr)

    history = model.fit(gen.flow(x_train, y_train, batch_size=bs, subset='training'),
                    validation_data=gen.flow(x_train, y_train, subset='validation'),
                    epochs=300,
                    verbose=1,
                    callbacks=callbacks,
                    class_weight=class_weight,
                   )

    model_name = f'star_galaxy_model_{exp_name}'
    model.save(os.path.join(paths['saved_model_path'],f"{model_name}.h5"))

    with open(os.path.join(paths['saved_model_path'],f"{model_name}.pkl"), 'wb') as f:
        pickle.dump(history.history, f)

    assert os.path.isfile(os.path.join(paths['saved_model_path'],f"{model_name}.h5")), "Saving the model failed miserably!"
    assert os.path.isfile(os.path.join(paths['saved_model_path'],f"{model_name}.pkl")), "Saving the history failed miserably!"
        
    return history
    
def model_evaluation(x_test: np.ndarray,
                     y_test: np.ndarray,
                     paths: list,
                     exp_name: str = "exp_0",
                     use_thresholding: bool = True,
                     ) -> tuple:
    
    # if true, use thresholding on the input images
    if use_thresholding:
        p_function = thresh_function
    else:
        p_function = lambda x: x
    
    # load saved model
    model_name = f'star_galaxy_{exp_name}'
    model_path = os.path.join(paths['saved_model_path'], f"{model_name}.h5")
    model = tf.keras.models.load_model(model_path)

    y_test = to_categorical(y_test)

    # apply preprocessing function
    x_test = [p_function(img) for img in x_test]
    x_test = np.asarray(x_test, dtype=np.float32)

    # get predictions
    y_pred = model.predict(x_test)

    # convert from categorical
    y_test = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_pred, axis=1)

    # compute metrics
    pre = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
        
    return (prec, rec, auc, f1)
