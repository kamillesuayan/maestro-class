import torch
import requests

test = ["ask server to evaluate the code", "get the score"]
test = test[
    0
]  # 0 checks the student ask for the server to evaluate their code; 1 gets the result from the server.


def asking(url, device, student_id=123):
    data = {
        "Application_Name": "Genetic_Attack",
        "data_type": "test",
        "id": student_id,
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
    url = "http://127.0.0.1:443"
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print("at test")
    student_id = 117036910009

    if test == "ask server to evaluate the code":
        asking(url, device, 11)
        asking(url, device, 11)
        asking(url, device, 11)
        asking(url, device, 11)
        asking(url, device, 11)
        asking(url, device, 11)
        asking(url, device, 11)
        asking(url, device, 22)
        asking(url, device, 33)
        asking(url, device, 44)
    elif test == "get the score":
        getScore(url, device, student_id)


if __name__ == "__main__":
    main()

