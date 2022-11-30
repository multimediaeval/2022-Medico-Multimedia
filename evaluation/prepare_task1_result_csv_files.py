from collections import defaultdict
import os
import csv
import glob
import argparse

from collections import defaultdict

import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input-dir", type=str)
parser.add_argument("-o", "--output-dir", type=str)

def read_csv(path):
    data = []
    with open(path, "r") as f:
        csv_reader = csv.reader(f, delimiter=",")
        header = next(csv_reader)
        for line in csv_reader:
            data.append(line)
    return np.array(data), header

if __name__ == "__main__":

    args = parser.parse_args()

    average_scores = {}

    for results_filepath in glob.glob(os.path.join(args.input_dir, "*", "*.csv")):

        run_parts = os.path.splitext(results_filepath)[0].split(os.sep)

        run_class_evaluated = run_parts[-1].split("_")[0]
        run_name = run_parts[-2]

        team_name = "-".join(run_name.split("-")[:-2])
        run_task1_type = run_name.split("-")[-2]

        run_id = run_name

        results_values, results_header = read_csv(results_filepath)
        
        if run_id not in average_scores:
            average_scores[run_id] = {
                "header": results_header,
                "firstcol": results_values[:, 0],
                "values": np.array(results_values[:, 1:], dtype=np.float32)
            }
        else:
            average_scores[run_id]["values"] += np.array(results_values[:, 1:], dtype=np.float32)

        team_output_path = os.path.join(args.output_dir, team_name, "task1", run_task1_type)

        if not os.path.exists(team_output_path):
            os.makedirs(team_output_path)

        with open(os.path.join(team_output_path, "%s-%s.csv" % (run_class_evaluated, run_name)), "w") as f:
            csv_writer = csv.writer(f, delimiter=",")
            csv_writer.writerow(results_header)
            csv_writer.writerows(results_values)

    print(average_scores.keys())

    for run_name, run_vales in average_scores.items():

        team_name = "-".join(run_name.split("-")[:-2])
        run_task1_type = run_name.split("-")[-2]

        team_output_path = os.path.join(args.output_dir, team_name, "task1", run_task1_type)

        with open(os.path.join(team_output_path, "average-%s.csv" % run_name), "w") as f:
            csv_writer = csv.writer(f, delimiter=",")
            csv_writer.writerow(run_vales["header"])
            values = []
            for val, col in zip((run_vales["values"] / 3).tolist(), run_vales["firstcol"].tolist()):
                values.append([col] + val)
            csv_writer.writerows(values)