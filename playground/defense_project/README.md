## Defence Project
**Important**: Before you start to run the code, download the cifar10 dataset for training, validating and testing from the link `https://canvas.eee.uci.edu/courses/43292/files/folder/data/defense_project` and move them under the path `playground/defense_project/datasets/CIFAR10/`.

In detail, you need to move three files `test_student_split.pt`, `train_student_split.pt`, `val_student_split.pt` to the path `playground/defense_project/datasets/CIFAR10/` and their sizes are around 44.5MB, 221.9MB and 221.5MB. If you meet with problems like `_pickle.UnpicklingError: invalid load key, 'v'.`, it means the data is loaded mistakenly and you need to check the dataset files.



Please revise the `tasks/defense_project/predict.py` and `tasks/defense_project/train.py` files to implement different defense methods.

You can use the following commands to evaluate the result: `python defense_project-Evaluator.py`.

The defense model wil be saved under `models/defense_project-model.pth`.

Please upload the defense model `models/defense_homework-model.pth` and the PY files `tasks/defense_project/predict.py` and `tasks/defense_project/train.py` to the gradescope by the name `model.pth`, `predict.py` and `train.py` without being zipped.
