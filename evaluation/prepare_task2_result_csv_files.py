#!/usr/bin/env python

import argparse
import os
import csv
import glob
import random
import warnings

from sklearn import metrics

import numpy as np

random.seed(0)
np.random.seed(0)

warnings.filterwarnings("ignore")

GROUND_TRUTH_PATH = ""
RESULTS_DIRECTORY = ""

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input-dir", type=str)
parser.add_argument("-g", "--ground-truth-filepath", type=str)
parser.add_argument("-o", "--output-dir", type=str)

def read_submission(submission_path):
    results = {}
    duplicates = {}
    lines = []
    with open(submission_path) as f:
        for row in csv.reader(f):
            lines.append(row)
            image_id = os.path.splitext(row[0].strip())[0].strip()

            if image_id in results:
                duplicates[image_id] = row[1].strip()
                continue
                
            results[image_id] = row[1].strip()
    return duplicates, np.array(lines), results

def read_csv(gt_path, variables):

    ground_truth = { }

    with open(gt_path) as csv_data:

        csv_reader = csv.reader(csv_data, delimiter=";")
        header = next(csv_reader)
        variable_indecies = [header.index(variable) for variable in variables]

        for row in csv_reader:
            video_id = os.path.splitext(row[0].strip())[0].strip()                
            ground_truth[ video_id ] = [ float(row[ variable_index ]) for variable_index in variable_indecies ]
            
    return ground_truth

def evaluate_submission(submission_path, ground_truth_filepath, output_dir):

    submission_attributes = submission_path.split(os.sep)
    
    team_name = submission_attributes[-4]
    task_name = submission_attributes[-3]
    run_id    = submission_attributes[-2]

    variables = [ "progressive_%", "non_progressive_%", "immotile_%" ]

    team_result_path = os.path.join(output_dir, team_name, task_name, run_id)

    if not os.path.exists(team_result_path):
        os.makedirs(team_result_path)

    gt_results = read_csv(ground_truth_filepath, variables)
    pred_results = read_csv(submission_path, variables)

    y_pred, y_truth = [], []

    for video_id, actual_class in gt_results.items():
        y_pred.append(gt_results[ video_id ])
        y_truth.append(pred_results[ video_id ])

    y_truth = np.array(y_truth)
    y_pred = np.array(y_pred)

    if len(y_truth) != len(y_pred):
        raise Exception("The number of predicted values is NOT equal to the ground truth!")

    mean_absolute_error = metrics.mean_absolute_error(y_truth, y_pred, multioutput="raw_values")
    mean_squared_error = metrics.mean_squared_error(y_truth, y_pred, multioutput="raw_values")
    root_mean_squared_error = metrics.mean_squared_error(y_truth, y_pred, multioutput="raw_values", squared=False)
    mean_squared_log_error = metrics.mean_squared_log_error(y_truth, y_pred, multioutput="raw_values")
    median_absolute_error = metrics.median_absolute_error(y_truth, y_pred, multioutput="raw_values")

    with open(os.path.join(team_result_path, "%s_%s_%s_metrics.csv" % (team_name, task_name, run_id)), "w") as f:
        f.write(",".join(["variable", "mean absolute error", "mean squared error", "root mean squared error", "mean squared log error", "median absolute error"]) + "\n")
        for index, variable_name in enumerate(variables):
            f.write(",".join([variable_name, str(mean_absolute_error[index]), str(mean_squared_error[index]), str(root_mean_squared_error[index]), str(mean_squared_log_error[index]), str(median_absolute_error[index])]) + "\n")
        f.write(",".join(["average", str(np.mean(mean_absolute_error)), str(np.mean(mean_squared_error)), str(np.mean(root_mean_squared_error)), str(np.mean(mean_squared_log_error)), str(np.mean(median_absolute_error))]))
        
if __name__ == "__main__":

    args = parser.parse_args()

    for team_submission_dir in glob.glob(os.path.join(args.input_dir, "*")):
        
        if "task2" not in os.listdir(team_submission_dir):
            continue

        for submission_filepath in glob.glob(os.path.join(team_submission_dir, "*", "*", "*.csv")):
            print("Evaluating %s..." % submission_filepath)
            evaluate_submission(submission_filepath, args.ground_truth_filepath, args.output_dir)