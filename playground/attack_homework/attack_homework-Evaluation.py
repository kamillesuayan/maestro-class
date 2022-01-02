import torch
import requests

test = ["ask server to evaluate the code", "get the score"]
test = test[
    0
]  # 0 checks the student ask for the server to evaluate their code; 1 gets the result from the server.


def asking(url, device, student_id=123, student_name="Alice"):
    data = {
        "Application_Name": "GeneticAttack",
        "data_type": "test",
        "id": student_id,
        "student_name": student_name,
        "task": "attack_homework",
    }
    final_url = "{0}/file_evaluator".format(url)
    response = requests.post(final_url, data=data)
    feedback = response.json()["feedback"]
    print(feedback)


def getScore(url, device, student_id=123):
    data = {
        "Application_Name": "GeneticAttack",
        "data_type": "test",
        "id": student_id,
        "task": "attack_homework",
    }
    final_url = "{0}/retrieve_result".format(url)
    response = requests.post(final_url, data=data)
    score = response.json()["score"]
    print(score)


def main():
    LOCAL = True
    if LOCAL == True:
        port = 5000
    else:
        port = 443

    url = "http://127.0.0.1:" + str(port)
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

