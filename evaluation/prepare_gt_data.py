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
parser.add_argument("-o", "--output-filepath", nargs="+")

def read_labels(path, image_index):
    data = defaultdict(list)
    with open(path, "r") as f:
        csv_reader = csv.reader(f, delimiter=" ")
        for line in csv_reader:
            data["all"].append([image_index, line[1], *line[2:6], -1, line[0], -1, -1])
            data[_CLASS_ID_MAPPING[int(line[0])]].append([image_index, line[1], *line[2:6], -1, 0, -1, -1])
    return data

if __name__ == "__main__":

    args = parser.parse_args()

    for input_dir, output_filepath in zip(args.input_dir, args.output_filepath):

        label_items = os.listdir(input_dir)

        for label_item in label_items:

            label_container = defaultdict(list)
            
            label_filepaths = list(glob.glob(os.path.join(input_dir, label_item, "labels", "*.txt")))
            label_filepaths = sorted(label_filepaths, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("_")[-1]))

            n_label_files = len(label_filepaths) 

            for label_filepath in label_filepaths:
                image_index = int(os.path.splitext(os.path.basename(label_filepath))[0].split("_")[-1])
                labels = read_labels(label_filepath, image_index)
                label_container["all"].extend(labels["all"])
                for class_id, class_name in _CLASS_ID_MAPPING.items():
                    label_container[class_name].extend(labels[class_name])

            for class_name, class_labels in label_container.items():

                run_name = os.path.basename(output_filepath)
                
                class_output_path = os.path.join(output_filepath + "-%s" % class_name, label_item, "gt", "gt.txt")
                
                if not os.path.exists(os.path.dirname(class_output_path)):
                    os.makedirs(os.path.dirname(class_output_path))

                with open(os.path.join(output_filepath + "-%s" % class_name, label_item, "seqinfo.ini"), "w") as f:
                    f.write("[Sequence]\n")
                    f.write("name=Medico\n")
                    f.write("imDir=img1\n")
                    f.write("frameRate=30\n")
                    f.write("seqLength=%i\n" % n_label_files)
                    f.write("imWidth=640\n")
                    f.write("imHeight=480\n")
                    f.write("imExt=.jpg")

                with open(class_output_path, "w") as f:
                    csv_writer = csv.writer(f, delimiter=" ") 
                    csv_writer.writerows(label_container[class_name])