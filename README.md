# Maestro

A service for educational purposes in the domain of adversarial attacks/defense.


## Structure overview

-   `Maestro/data/`
	-   Handles dataset loading.
	-   Contains wrapper for huggingface datasets.
	-   Wrapper for Torchvision datasets
-   `Maestro/evaluator/`

	-   Evaluator class to evaluate different applications
	-   Compute attack rate/constraint violations etc.
-   `Maestro/constraint/`
	-   Contains class for different constraints.
-   `Maestro/attacker_helper/`
	-   Contains helper file for the attacker to query the server
-   `Maestro/examples/`

    - Examples for loading custom models and datasets (good for getting around NLP libraries)
	-   Examples for several scenarios (outdated, pre REST API examples)
	-   Server Examples (example code using REST API)
	-   Attacker File (contains starting files for the attacker)
	-   Evaluation (contains evaluations for the attacker file as well as sample complete attacker files)


-   `Maestro/models/`

	-   Handles model loading (from HugginFace)

	-   A couple customized models such as LSTM


-   `Maestro/pipeline/`

	-   Contains AutoPipeline,Pipeline, and Model_Wrapper. Crucial logics from the backend of the server.


-   `Maestro/server/`

	-   Handles the flask api server

	-   Complete the methods that handle POST requests

-   `Maestro/utils/`
	- Utility functions such as move_to_device

## How to use
### Server Side
**Create a virtual enviroment with either conda or vm. Set the python version to 3.7**

If you are using Mac OS, to install Python 3.7.5 use:
```
$ brew install pyenv
```
Then add the following lines in your `.bash_profile` (or alternative file).
```
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```
Reload the shell with `exec "$SHELL"`. Then, install the desired python version:
```
$ pyenv install 3.7.5
$ pyenv global 3.7.5
```
If the above command fails, use a patch:
```
$ CFLAGS="-I$(brew --prefix openssl)/include -I$(brew --prefix bzip2)/include -I$(brew --prefix readline)/include -I$(xcrun --show-sdk-path)/usr/include" LDFLAGS="-L$(brew --prefix openssl)/lib -L$(brew --prefix readline)/lib -L$(brew --prefix zlib)/lib -L$(brew --prefix bzip2)/lib" pyenv install --patch 3.7.5 < <(curl -sSL https://github.com/python/cpython/commit/8ea6353.patch\?full_index\=1)
$ pyenv global 3.7.5
```
Make sure the right version is installed with `python3 --version`.

**Install Maestro:**
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

**Run a local server:**
Get into the server folder and run the Flask application:
```
$ (env) cd Maestro/server
$ (env) python3 app.py
```

If you need to run task queue, open 2 more terminals and run:
```
./run-redis.sh # run the redis database
celery -A app.celery worker # run the celery workers
celery -A app.celery worker -P solo
```

Change or update `app.py` and `model.py` accordingly for what models to load and how to load models.


### Evalutor tutorial
```
python ./Maestro/server/app.py # run the server
# upload the file to ./Maestro/tmp/attack_homework/GeneticAttack_117036910009.py
python examples/Evaluations/CV/Attack-Homework/Genetic_Algorithm_Attack_Evaluation.py # change the line 4: test = test[0] # 0 checks the student ask for the server to evaluate their code; 1 gets the result from the server.

```

### To Run Task Queue
```
Go to the Maestr/server/ directory
python app.py # run the server
sh run-redis.sh # run redis server
celery -A app.celery worker # intitialize the worker

```
