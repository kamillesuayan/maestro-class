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
**Install Maestro:**
```
pip install -r requirements.txt
```
**Run a local server:**
```
python -m pip install -e .
```
```
cd Maestro/server
```
```
python app.py
```
change or update `app.py` and `model.py` accordingly for what models to load and how to load models.

### Attacker Side
Text:
| Application  | Evaluation | Constraints 
| ------------- | ------------- | ------------- | 
| `Hotflip`  | Untargeted Label Flip Rate  | Number of tokens flipped  | 
| `Universal Triggers`  | Untargeted Label Flip Rate  | Length of trigger  | 

Image:
| Application  | Evaluation | Constraints
| ------------- | ------------- | ------------- | 
| `FGSM`  | Untargeted Label Flip Rate  | Within Epsilon Ball | 
| `Data Poisoning`  | Targeted Label Flip Rate  | Number of Data Poisoned  | 

Use files in `Maestro/examples/Attacker_File` to implment the attack. Then put the finished file in the same folder as the evaluation file and then call evaluation:
```
python FGSM_Eval.py
```

### Some Issues
1. If some problems that are similar to `The current PyTorch install supports CUDA capabilities sm_37 sm_50 sm_60 sm_70 sm_75.` appear, check the [link](https://pytorch.org/get-started/locally/) to reinstall to suitable pytorch version.


## TODOs
- add full API support for `Data Poisoning`,currently it can only run locally. 
- text version of `Data Poisoning`
