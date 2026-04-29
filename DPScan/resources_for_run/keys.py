Uniapi = False # If you use Gemini API, please set it as False

GeminiKey = [
    # "If you use Gemini API, put your key here."
]

UniapiPaidKey = [
    # "If you use UniAPI, put your key here."
    # "However, considering that LLMRAG module also needs a Gemini API Key to run, make sure that you have at least one Gemini key."
]

from google import genai
from google.genai import types

if __name__ == '__main__':
    client = genai.Client(api_key="xxx", http_options={'timeout': 600000})
    models = client.models.list()
    for m in models:
        print(m.name)
