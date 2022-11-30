python TrackEval/scripts/run_medico_detection.py --CLASSES_TO_EVAL "sperm" --SPLIT_TO_EVAL "detection-test-sperm" --IGNORE_TRACK_IDS True;
python TrackEval/scripts/run_medico_detection.py --CLASSES_TO_EVAL "cluster" --SPLIT_TO_EVAL "detection-test-cluster" --IGNORE_TRACK_IDS True;
python TrackEval/scripts/run_medico_detection.py --CLASSES_TO_EVAL "small-or-pinhead" --SPLIT_TO_EVAL "detection-test-small-or-pinhead" --IGNORE_TRACK_IDS True;

python TrackEval/scripts/run_medico_tracking.py --CLASSES_TO_EVAL "sperm" --SPLIT_TO_EVAL "tracking-test-sperm";
python TrackEval/scripts/run_medico_tracking.py --CLASSES_TO_EVAL "cluster" --SPLIT_TO_EVAL "tracking-test-cluster";
python TrackEval/scripts/run_medico_tracking.py --CLASSES_TO_EVAL "small-or-pinhead" --SPLIT_TO_EVAL "tracking-test-small-or-pinhead";

python prepare_task1_result_csv_files.py;