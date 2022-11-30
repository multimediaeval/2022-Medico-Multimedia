from collections import defaultdict
import os
import csv
import glob
import argparse

parser = argparse.ArgumentParser()

_CLASS_ID_MAPPING = {
    0: "sperm",
    1: "small-or-pinhead",
    2: "cluster"
}

parser.add_argument("-i", "--input-dir", nargs="+")
parser.add_argument("-d", "--output-dir", type=str)
parser.add_argument("-o", "--output-filepath", nargs="+")

def read_labels(path, image_index):
    global TRACKER_COUNTER
    data = defaultdict(list)
    with open(path, "r") as f:
        csv_reader = csv.reader(f, delimiter=" ")
        for line in csv_reader:
            data["all"].append([image_index, 1, *line[2:6], -1, line[0], -1, -1])
            data[_CLASS_ID_MAPPING[int(line[0])]].append([image_index, 1, *line[2:6], -1, 0, -1, -1])
            TRACKER_COUNTER += 1
    return data

if __name__ == "__main__":

    args = parser.parse_args()

    for input_dir, team_run_name in zip(args.input_dir, args.output_filepath):

        label_items = os.listdir(input_dir)

        for label_item in label_items:
            
            label_container = defaultdict(list)
            
            label_filepaths = list(glob.glob(os.path.join(input_dir, label_item, "labels", "*.txt")))
            label_filepaths = sorted(label_filepaths, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("_")[-1]))

            TRACKER_COUNTER = 0
            
            for label_filepath in label_filepaths:
                image_index = int(os.path.splitext(os.path.basename(label_filepath))[0].split("_")[-1]) + 1
                labels = read_labels(label_filepath, image_index)
                label_container["all"].extend(labels["all"])
                for class_id, class_name in _CLASS_ID_MAPPING.items():
                    label_container[class_name].extend(labels[class_name])

            for class_name, class_labels in label_container.items():

                run_name = os.path.basename(args.output_dir)
                
                class_output_path = os.path.join(args.output_dir + "-%s" % class_name, team_run_name, "data", "%s.txt" % label_item)

                if not os.path.exists(os.path.dirname(class_output_path)):
                    os.makedirs(os.path.dirname(class_output_path))

                with open(class_output_path, "w") as f:
                    csv_writer = csv.writer(f, delimiter=" ") 
                    csv_writer.writerows(label_container[class_name])