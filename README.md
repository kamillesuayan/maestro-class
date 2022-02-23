## Reminder

- As to the defence homework, refer to the `playground/defense_homework/README.md`.
- As to the defence project, refer to the `playground/defense_project/README.md`.


# Maestro

A service for educational purposes in the domain of adversarial attacks / defense.

## Structure overview

-   `Maestro/data/`
	-   Handles dataset loading.
	-   Contains wrapper for huggingface datasets.
	-   Wrapper for Torchvision datasets
-   `Maestro/evaluator/`
	-   Evaluator class to evaluate different applications.
	-   Compute attack rate/constraint violations etc.
-   `Maestro/attacker_helper/`
	-   Contains helper file for the attacker to query the server.
-   `Maestro/models/`
	-   Handles model loading (from HugginFace).
	-   A couple customized models such as LSTM.
-   `Maestro/pipeline/`
	-   Contains AutoPipeline, Pipeline, and Model_Wrapper. Crucial logics from the backend of the server.
-   `Maestro/server/`
	-   Handles the flask api server.
	-   Complete the methods that handle POST requests.
-   `Maestro/utils/`
	- Utility functions such as move_to_device.
-   `playground`
	-   **This is the folder where students will write their own code.**
	-   Attacker File (contains starting files for the attacker).
	-   Evaluation (contains evaluations for the attacker file as well as sample complete attacker files).

## How to use
### Server Side
**Create a virtual enviroment with either conda or vm. Set the python version to 3.9.7**. Make sure the right version is installed with `python3 --version`.

#### Install Maestro
Make sure you have the virtual environment set (at the root folder). You can use any virtual environment, for instance [venv](https://docs.python.org/3/tutorial/venv.html) or [anaconda](https://docs.anaconda.com/anaconda/install/index.html). Example:
```
$ python3 -m venv env
$ source env/bin/activate
```
Once the virtual environment is set, install the requirements:
```
$ (env) pip3 install -r requirements.txt
```
Finally, install Maestro:
```
$ (env) python3 -m pip install -e .
```

#### Run a local server
##### Running the app
Get into the server folder and run the Flask application:
```
$ (env) cd Maestro/server
$ (env) python3 app.py
```
If you happen to have problems with `tkinter` on Mac OS, for instance, an error of the kind: `ModuleNotFoundError: No module named '_tkinter'`, install the module with `brew`:
```
$ (env) brew install python-tk
```

##### Customize the app
Depending on the assignment you need to do, you will need to change or update `app.py` and `model.py` accordingly for what models to load and how to load models. That is easy! Here are the steps:

- The first lines after local imports in `app.py` we have the paths for 4 the configuration files (2 for assignments and 2 for projects). Just uncomment the one you need and comments the others:
```
application_config_file = "Server_Config/Genetic_Attack.json"    # (Assignment 1)
# application_config_file = "Server_Config/Attack_Project.json"  # (Project 1)
# application_config_file = "Server_Config/Adv_Training.json"    # (Assignment 2)
# application_config_file = "Server_Config/Defense_Project.json" # (Project 2)
```
**If you cannot run `app.py` check that you are passing the right configuration file!**

### Code your assignment
The naming convention for submission files is: `<task>-<id>-<name>.py`. An example would be `attack_project-110-Team Fire.py` or `attack_homework-11-Alice.py`.
- Go to the `playground` folder. There you will find a folder for each assignment/project with the template and evaluator file.
- Fill in the TODO sections in the corresponding template file and follow the conventions in naming the file.
- Change the last lines in the evaluation file to match your name and UCI ID.

### Evaluator tutorial
To evaluate your coded solution, open a new terminal tab while running the `app.py` and do:
- Activate the virtual environment as always.
- In the evaluation script (for attack it is `attack_homework-Evaluation.py`) make sure the 4th line is set to `test[0]`.
- Run the evaluation script of the corresponding assignment. Example for attack homework:
```
$ cd playground/attack_homework
$ (env) python3 attack_homework-Evaluation.py
```
After that, a request will be sent to the server, you will be able to see the output in the other terminal tab, where `app.py` is running.
- You can see the saved outputs (plots) in the `Maestro/server` folder and the recording (final success rate) in the corresponding assignment folder. In the case of attack homework, after a successful execution you will be able to compare the images before and after genetic attack. Those images will be saved as `before_GA.png` and `after_GA.png` in the `Maestro/server` folder.

### Running FGSM attack
After importing latest updates, first uncomment the second line([#L28](https://github.com/ucinlp/maestro-class/blob/main/Maestro/server/app.py#L28)) in application configurations in app.py
```
# application_config_file = "Server_Config/Genetic_Attack.json"    # (Assignment 1)
application_config_file = "Server_Config/FGSM_Attack.json"
# application_config_file = "Server_Config/Attack_Project.json"  # (Project 1)
# application_config_file = "Server_Config/Adv_Training.json"    # (Assignment 2)
# application_config_file = "Server_Config/Defense_Project.json" # (Project 2)
```
Run the app (as mentioned in "Running the app" above)

To the attack file, Activate the virtual environment as always and run the following commands
```
$ cd playground/attack_homework
$ (env) python3 attack_homework-FGSM_example.py
```
You can playaround this attack by changing the value of epsilon ([#L93](https://github.com/ucinlp/maestro-class/blob/main/playground/attack_homework/attack_homework_FSGM_example.py#L93)).
```
  epsilon = 0.214
```
