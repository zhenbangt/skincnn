from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

IMG_WIDTH, IMG_HEIGHT = 224, 224

import tensorflow_hub as hub
from tensorflow.keras.applications.densenet import (
    DenseNet121,
    preprocess_input,
)
import pandas as pd
import numpy as np
import os
import IPython.display as display
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from tensorflow.keras.layers import (
    Dense,
    GlobalAveragePooling2D,
    Conv2D,
    Flatten,
    GlobalMaxPooling2D,
    Dropout,
)

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, Nadam
from datetime import datetime
from packaging import version
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import (
    TensorBoard,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
import efficientnet.tfkeras as enet


gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(
            len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs",
        )
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
else:
    print("No GPU found!")
    quit()


def append_extension(fn):
    return (fn + ".jpg").zfill(7)


def ordered_logit(class_number):
    # zero portability
    target = np.zeros(4, dtype=int)
    target[: class_number - 2] = 1
    return target


DATADIR = r"./adult"
CSV_PATH = r"./adult/CastControls_ALP.xlsx"
response = pd.read_excel(CSV_PATH, sheet_name=0,)[["GreenID", "Grade"]].dropna(
    axis=0, subset=["Grade"]
)
response.Grade = response.Grade.astype("int")
response.GreenID = response.GreenID.astype("str").apply(append_extension)
response = response[response.Grade != 99]
response = pd.concat(
    [response, pd.DataFrame.from_dict(dict(response.Grade.apply(ordered_logit))).T,],
    axis=1,
)


# shuffle dataset
response = response.sample(frac=1)
seed = np.random.randint(30027)


def soft_acc(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))


def soft_acc_multi_output(y_true, y_pred):
    return K.mean(
        K.all(
            K.equal(
                K.cast(K.round(y_true), "int32"), K.cast(K.round(y_pred), "int32"),
            ),
            axis=1,
        )
    )

# from tensorflow.keras import mixed_precision

# policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
# mixed_precision.experimental.set_policy(policy)



def generate_train_val_test(train_index, val_index, test_index):
    train_dataset = response.iloc[train_index]
    val_dataset = response.iloc[val_index]
    test_dataset = response.iloc[test_index]
    train_gen = ImageDataGenerator(
        rotation_range=15,
        fill_mode="reflect",
        horizontal_flip=True,
        rescale=1.0 / 255.0,
        zoom_range=0.1,
    )
    valid_test_gen = ImageDataGenerator(rescale=1.0 / 255.0,)

    train_set = train_gen.flow_from_dataframe(
        dataframe=train_dataset,
        directory=DATADIR,
        x_col="GreenID",
        target_size=(380, 380),
        color_mode="rgb",
        subset="training",
        shuffle=True,
        y_col=[0, 1, 2, 3,],
        class_mode="raw",
    )

    validation_set = valid_test_gen.flow_from_dataframe(
        dataframe=val_dataset,
        directory=DATADIR,
        x_col="GreenID",
        target_size=(380, 380),
        color_mode="rgb",
        subset="training",
        shuffle=False,
        batch_size=64,
        y_col=[0, 1, 2, 3,],
        class_mode="raw",
    )

    test_set = valid_test_gen.flow_from_dataframe(
        dataframe=test_dataset,
        directory=DATADIR,
        x_col="GreenID",
        target_size=(380, 380),
        color_mode="rgb",
        subset="training",
        shuffle=False,
        batch_size=64,
        y_col=[0, 1, 2, 3,],
        class_mode="raw",
    )
    return train_set, validation_set, test_set


from sklearn.model_selection import StratifiedKFold

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
innerkf = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
response = response.sample(frac=1.0)
early_stopping = EarlyStopping(
    monitor="val_loss", patience=50, restore_best_weights=True,
)
reduce_lr_plateau = ReduceLROnPlateau(monitor="val_loss", patience=7, factor=0.8)


def generate_base_model():
    conv_base = enet.EfficientNetB4(
        include_top=False, input_shape=(380, 380, 3), pooling="avg", weights="noisy-student",
    )
    conv_base.trainable = False

    x = conv_base.output
    x = Dropout(0.8)(x)
    preds = Dense(4, activation="sigmoid")(x)
    model = Model(inputs=conv_base.input, outputs=preds)

    model.compile(
        optimizer=keras.optimizers.Nadam(),
        loss="binary_crossentropy",
        metrics=[soft_acc_multi_output],
    )

    return model


def fine_tune_model(model, fine_tune=None):
    if fine_tune is None:
        try:
            fine_tune = [layer.name for layer in model.layers].index(r"top_conv")
        except:
            pass

    model.trainable = True
    for layer in model.layers[:fine_tune]:
        layer.trainable = False
    for layer in model.layers[fine_tune:]:
        layer.trainable = True

    model.compile(
        optimizer=keras.optimizers.Nadam(),
        loss="binary_crossentropy",
        metrics=[soft_acc_multi_output],
    )

    return model


import gc
def stratified_cv(fine_tune_layer=None):
    acc_coef_scores = []
    raw_outputs = []
    for train_index, val_test_index in kf.split(
        np.zeros(len(response)), response["Grade"]
    ):
        val_index, test_index = next(
            innerkf.split(
                np.zeros(len(val_test_index)), response["Grade"].iloc[val_test_index]
            )
        )
        val_index, test_index = val_test_index[val_index], val_test_index[test_index]
        train_set, validation_set, test_set = generate_train_val_test(
            train_index, val_index, test_index
        )
        model = generate_base_model()

        _ = model.fit(
            x=train_set,
            epochs=10,
            validation_data=validation_set,
            callbacks=[early_stopping, reduce_lr_plateau],
            verbose=0,
        )

        model = fine_tune_model(model, fine_tune=fine_tune_layer)

        _ = model.fit(
            x=train_set,
            epochs=200,
            validation_data=validation_set,
            callbacks=[early_stopping, reduce_lr_plateau],
            verbose=0,
        )

        batch = next(test_set)
        true_labels = batch[1]
        predictions = model.predict(batch[0])
        acc = soft_acc_multi_output(predictions, true_labels).numpy()
        corr = np.corrcoef(np.sum(predictions, axis=1), np.sum(true_labels, axis=1))[0][
            1
        ]
        acc_coef_scores.append([acc, corr])
        raw_outputs.append([np.array(response.iloc[test_index].index), true_labels, predictions])
        del train_set, validation_set, test_set, _, model, batch, true_labels, predictions, acc, corr
        tf.keras.backend.clear_session()
        gc.collect()
    return acc_coef_scores, raw_outputs


#  np.where(np.array(['conv' in layer.name for layer in model.layers]) == True)[0][::-1]
trainable_sequence = np.array([469, 464, 460, 452, 449, 447, 439, 436, 432, 424, 421, 417, 409, 406,
       402, 394, 391, 387, 379, 376, 372, 364, 361, 357, 349, 346, 342,
       334, 331, 329, 321, 318, 314, 306, 303, 299, 291, 288, 284, 276,
       273, 269, 261, 258, 254, 246, 243, 241, 233, 230, 226, 218, 215,
       211, 203, 200, 196, 188, 185, 181, 173, 170, 166, 158, 155, 153,
       145, 142, 138, 130, 127, 123, 115, 112, 108, 100,  97,  95,  87,
        84,  80,  72,  69,  65,  57,  54,  50,  42,  39,  37,  29,  26,
        22,  14,  12,   4,   1])

# any trainable sequence
# trainable_sequence = np.array([469, 465, 464, 461, 460, 458, 457, 453, 452, 450, 449, 448, 447,
#        445, 444, 440, 439, 437, 436, 433, 432, 430, 429, 425, 424, 422,
#        421, 418, 417, 415, 414, 410, 409, 407, 406, 403, 402, 400, 399,
#        395, 394, 392, 391, 388, 387, 385, 384, 380, 379, 377, 376, 373,
#        372, 370, 369, 365, 364, 362, 361, 358, 357, 355, 354, 350, 349,
#        347, 346, 343, 342, 340, 339, 335, 334, 332, 331, 330, 329, 327,
#        326, 322, 321, 319, 318, 315, 314, 312, 311, 307, 306, 304, 303,
#        300, 299, 297, 296, 292, 291, 289, 288, 285, 284, 282, 281, 277,
#        276, 274, 273, 270, 269, 267, 266, 262, 261, 259, 258, 255, 254,
#        252, 251, 247, 246, 244, 243, 242, 241, 239, 238, 234, 233, 231,
#        230, 227, 226, 224, 223, 219, 218, 216, 215, 212, 211, 209, 208,
#        204, 203, 201, 200, 197, 196, 194, 193, 189, 188, 186, 185, 182,
#        181, 179, 178, 174, 173, 171, 170, 167, 166, 164, 163, 159, 158,
#        156, 155, 154, 153, 151, 150, 146, 145, 143, 142, 139, 138, 136,
#        135, 131, 130, 128, 127, 124, 123, 121, 120, 116, 115, 113, 112,
#        109, 108, 106, 105, 101, 100,  98,  97,  96,  95,  93,  92,  88,
#         87,  85,  84,  81,  80,  78,  77,  73,  72,  70,  69,  66,  65,
#         63,  62,  58,  57,  55,  54,  51,  50,  48,  47,  43,  42,  40,
#         39,  38,  37,  35,  34,  30,  29,  27,  26,  23,  22,  20,  19,
#         15,  14,  13,  12,  10,   9,   5,   4,   2,   1])

from tqdm import tqdm

fine_tune_scores_acc_coef = []
fine_tune_raw_outputs = []

for i in tqdm(range(45,60)):
    fine_tune = trainable_sequence[i]
    acc_coef_scores, raw_outputs = stratified_cv(fine_tune)
    fine_tune_scores_acc_coef.append(acc_coef_scores)
    fine_tune_raw_outputs.append(raw_outputs)
    np.save(r"./stratified_cross_validation_results/effB4/multinomial_acc_coef_45-60", np.array(fine_tune_scores_acc_coef))
    np.save(r"./stratified_cross_validation_results/effB4/multinomial_raw_outputs_45-60", np.array(fine_tune_raw_outputs))
    del acc_coef_scores, raw_outputs
    tf.keras.backend.clear_session()
    gc.collect()
