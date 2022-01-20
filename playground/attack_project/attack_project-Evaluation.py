import torch
import requests
import argparse
test = ["ask server to evaluate the code", "get the score"]
test = test[0] # 0 checks the student ask for the server to evaluate their code; 1 gets the result from the server.

def asking(url, device, student_id=123, student_name="Alice"):
    data = {"Application_Name": "Project_Attack", "data_type": "test", "id": student_id, "student_name": student_name,"task": "attack_project"}
    final_url = "{0}/file_evaluator".format(url)
    response = requests.post(final_url, data=data)
    feedback = response.json()["feedback"]
    print(feedback)


def getScore(url, device, student_id=123):
    data = {"Application_Name": "Project_Attack", "data_type": "test", "id": student_id, "task": "attack_project"}
    final_url = "{0}/evaluate_result".format(url)
    response = requests.post(final_url, data=data)
    score = response.json()
    print(score)

def main():
    # command line: python3 attack_project-Evaluation.py --local OR python3 attack_project-Evaluation.py --remote to test on the respective server
    parser = argparse.ArgumentParser(description="Attack Project Evaluation")
    parser.add_argument('--remote', dest='feature', action='store_true')
    parser.add_argument('--local', dest='feature', action='store_false')
    parser.set_defaults(feature=True)
    args = parser.parse_args()
    REMOTE = args.feature
    if REMOTE == True:
        ip = "128.195.151.199:"
        port = 443
    else:
        ip = "127.0.0.1:"
        port = 5000

    url = "http://" + ip + str(port)
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print("at test")
    student_id = 11
    student_name = "Alice"



    if test == "ask server to evaluate the code":
        asking(url, device, student_id, student_name)
    elif test == "get the score":
        getScore(url, device, student_id)




if __name__ == "__main__":
    main()

