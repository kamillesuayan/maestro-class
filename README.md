# Maestro

A service for educational purposes in the domain of adversarial attacks/defense. 


## Structure overview

-   `Maestro/data/`
    
	-   Handles dataset loading.
    
	-   Contains wrapper for huggingface datasets.
    
	-   Wrapper for Torchvision datasets
    

-   `Maestro/examples/`
    
    - Examples for loading custom models and datasets (good for getting around NLP libraries)
	-   Examples for several scenarios (outdated, pre REST API examples) 
	-   Server Examples (example code using REST API)
    

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
- `docker/`
	- docker setup for the attackers, currently have two examples: Hotflip and Universal Triggers
## How to use
### Server Side
**Install Maestro:**
```
pip install -r requirements.txt
```
**Run a local server:**
```
cd Maestr/server
```
```
python app.py
```
change or update `app.py` and `model.py` accordingly for what models to load and how to load models.

### Attacker Side
To create a docker file for the attackers, first you need to install docker https://docs.docker.com/engine/install/
Then:
```
cd docker/Hotflip
```
```
docker build --tag docker-maestro .
```
This creates a docker image which you can distribute maybe via docker hub.
Note the file `attack.py` contains the code that needs to be implemented by the attacker. This file contains a method that takes in original data, and output perturbed data.
During evaluation at the server's side, we can directly run the container like this:
```
docker run -it --rm -v %cd%/data.pkl:/app/data.pkl docker-maestro
```
where data.pkl is the cutomized data that we have but attackers don't. The `-v` option mounts the data.pkl inside the docker container.

## TODOs
- get_batch_output needs to support both NLP and CV (currently separated)
- the model wrapper has a downfall where only one application can exist, it keeps adding methods to the old object. change this completely
- The attacker docker is using the cpu verson of torch. Using GPU would complicate the matter quite a lot. Does using just the cpu matter that much if we control the scale of the problem?
	- since most curcial operations are done 
