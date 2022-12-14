import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, auc, f1_score

from utils import get_data
from model import model_training, model_evaluation


def run_experiment(hyperparameters: dict) -> tuple:

    paths = {'model_path' : os.path.join('./models'), 
             'weights_path' : os.path.join('./models','weights'),
             'saved_model_path' : os.path.join('./models', 'saved_model'),
             'checkpoint_path' : os.path.join('./models','checkpoints'),
             'data_path' : '../Cutout Files/',
            }

    for key, value in paths.items():
        if not os.path.exists(value):
            os.mkdir(value)

    beta_token = int(hyperparameters["beta"]*100)
    thr_token = str(hyperparameters["use_thresholding"]).lower()
    backbone_token = hyperparameters["backbone_name"]
    optim_token = hyperparameters["optim_name"]
    layer_token = hyperparameters["trainable_layers"]

    exp_name = f"{backbone_token}_{optim_token}_{layer_token}_{beta_token}_{thr_token}"

    # load data
    data, labels = get_data(paths["data_path"])

    # train-test split with stratification
    # random state is fixed for reproducibility
    x_train, x_test, y_train, y_test = train_test_split(data, 
                                                        labels, 
                                                        test_size=0.2,
                                                        random_state=111, 
                                                        shuffle=True,
                                                        stratify=labels)

    history = model_training(x_train,
                             y_train,
                             paths,
                             exp_name,
                             **hyperparameters,
                            )

    prec, rec, auc, f1 = model_evaluation(x_test, 
                                          y_test, 
                                          paths, 
                                          exp_name, 
                                          hyperparameters["use_thresholding"])

    out_string = f"Precision = {prec}\nRecall = {rec}\nAUC = {auc}\nF1-score = {f1}"
    print(out_string)

    out_path = os.path.join(paths["saved_model_path"], f"{exp_name}.txt")

    with open(out_path, 'w') as f:
        f.write(out_string)
    
    return history, prec, rec, auc, f1

def main():

    #HYPERPARAMETERS
    hyperparameters = {
    "rotation_range" : 20,
    "w_init" : 'imagenet',
    "trainable_layers" : 5,
    "lr" : 0.001,
    "bs" : 128,
    "mean_ovs" : 10,
    "backbone_name" : "EfficientNetB0",
    "optim_name" : "Adam",
    "beta" : 0., # parameter for effective number of samples weighting
    "use_thresholding" : True,
    }

    output = {}
    for key in hyperparameters:
        output[key] = []
    output["pre"] = []
    output["rec"] = []
    output["auc"] = []
    output["f1"] = []

    decision_steps = [3, 6, 9, 11, 16] # choose best configuration after <decision_steps> experiments

    experimental_config = [
        ("backbone_name", "VGG16"),
        ("backbone_name", "MobileNetV2"),
        ("backbone_name", "EfficientNetB0"),
        ("backbone_name", "DenseNet121"),
        ("trainable_layers", 3),
        ("trainable_layers", 5),
        ("trainable_layers", 7),
        ("optim_name", "Adam"),
        ("optim_name", "RMSprop"),
        ("optim_name", "SGD"),
        ("use_thresholding", False),
        ("use_thresholding", True),
        ("beta", 0.),
        ("beta", 0.9),
        ("beta", 0.99),
        ("beta", 0.999),
    ]

    dec_counter = 0
    prev_idx = 0
    best_indices = []

    for n, param in enumerate(experimental_config):

        if n == decision_steps[dec_counter]:
            # restore best parameters found before
            best_index = int(np.where(output["f1"] == np.max(output["f1"][prev_idx:]))[0][0])
            best_indices.append(best_index)
            for idx in best_indices:
                best_key, best_val = experimental_config[idx]
                print(f"Best parameter {best_key}: {best_val}")
                hyperparameters[best_key] = best_val
            
            prev_idx = n + 1

            dec_counter += 1

        key, val = param
        hyperparameters[key] = val

        for key in hyperparameters:
            output[key].append(hyperparameters[key])

        history, prec, rec, auc, f1 = run_experiment(hyperparameters)

        output["pre"].append(prec)
        output["rec"].append(rec)
        output["auc"].append(auc)
        output["f1"].append(f1)

    output_frame = pd.DataFrame(data=output)
    output_frame.to_csv("all_results.csv", sep=';', encoding='utf-8')
    
if __name__ == "__main__":
    main()
    print("The experiments have been finished!")
