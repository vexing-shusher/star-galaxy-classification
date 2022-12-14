import os
import argparse
import numpy as np
import pandas as pd

from distutils.util import strtobool

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

def main(args):

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
    
    if os.path.exists("./all_results.csv"):
        old_frame = pd.read_csv("./all_results.csv", sep=';', encoding='utf-8')
        
        # find best configuration
        f1s = old_frame["f1"].to_numpy()
        best_idx = np.where(f1s == f1s.max())[0][0]
        
        # set hyperparameters to the best configuration
        for key in hyperparameters:
            hyperparameters[key] = old_frame[key][best_idx]
            
    types_dict = {"int":int, "float":float, "bool":strtobool, "str":str}
    
    # set the chosen hyperparameter to the chosen value (allowed types: string, float, int, bool)
    hyperparameters[args.par_name] = types_dict[args.type](args.par_val)

    for key in hyperparameters:
        output[key].append(hyperparameters[key])
        
    # run the experiment
    history, prec, rec, auc, f1 = run_experiment(hyperparameters)

    output["pre"].append(prec)
    output["rec"].append(rec)
    output["auc"].append(auc)
    output["f1"].append(f1)

    output_frame = pd.DataFrame(data=output)
    
    # append the new results to the existing report
    if os.path.exists("./all_results.csv"):
        output_frame = pd.concat([old_frame, output_frame], ignore_index=True)
    
    # drop extra columns that arise due to concatenation
    columns_to_drop = [col for col in output_frame.columns if col.split(' ')[0] == "Unnamed:"]
    output_frame.drop(columns_to_drop, axis=1, inplace=True)

    output_frame.to_csv("all_results.csv", sep=';', encoding='utf-8')
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", 
                        required=True, 
                        type=str, 
                        choices=["int", "float", "bool", "str"])
    parser.add_argument("--par_name", 
                        required=True, 
                        type=str,
                        choices=["backbone_name", "optim_name", "trainable_layers", "use_thresholding", "beta"])
    parser.add_argument("--par_val", required=True)
    
    args = parser.parse_args()
    
    main(args)
    print("The experiment has been finished!")