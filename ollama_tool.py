import requests



def generate_report(prompt):


    response = requests.post(
        "http://localhost:11434/api/generate",

        json={
            "model": "phi3",
            "prompt": prompt,
            "stream": False,
            "options":{
               "temperature":0.2
        }
        }
    )


    if response.status_code == 200:

        return response.json()["response"]

    else:

        return "Ollama error: " + response.text