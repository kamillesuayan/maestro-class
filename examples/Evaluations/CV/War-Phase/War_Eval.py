from Maestro.attacker_helper.attacker_request_helper import virtual_model

# 1 Defense Eval
# First we generate the attacked dataset using Maestro method
# If already exist, just load it

url = "http://127.0.0.1:5000"
application_name = "FGSM"
vm = virtual_model(url, application_name=application_name)
# dataset = vm.get_data(data_type="test", perturbation="FGSM")
# print(len(dataset))
# print(dataset[0])

# Call corresponding evaluator

# 2 Attack Eval
# Let Maestro run a defeneded model
print("Switch Weights of the App")
vm.switch_weights(
    file_path="/home/junliw/Maestro_Project/Maestro/examples/Evaluations/CV/Defense-Project/junlin_group_project/lenet_defended_model.pth"
)
