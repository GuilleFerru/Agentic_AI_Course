import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from IPython.display import Markdown, display

load_dotenv(override=True)

# API Keys
openai_api_key = os.getenv('OPENAI_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')
deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')

# Check API keys
api_keys = {
    "openai": openai_api_key,
    "google": google_api_key, 
    "deepseek": deepseek_api_key,
    "groq": groq_api_key
}

for name, key in api_keys.items():
    status = "exists" if key else "not set"
    print(f"API Key for {name}: {status}")

print("--------------------------------")

def get_llm_response(model_name, messages, api_key=None, base_url=None):
    """Simple function to get response from any OpenAI-compatible API"""
    if base_url:
        client = OpenAI(api_key=api_key, base_url=base_url)
    else:
        client = OpenAI(api_key=api_key)
    
    response = client.chat.completions.create(model=model_name, messages=messages)
    return response.choices[0].message.content

# Cordinator: ejecuta la pregunta en cada modelo listado y recoge las respuestas
def coordinator_llm_responses(question, models):
    responses = {}
    messages = [{"role": "user", "content": question}]
    for model_name, api_key, base_url in models:
        if api_key:
            try:
                response = get_llm_response(model_name, messages, api_key, base_url)
                responses[model_name] = response
            except Exception as e:
                responses[model_name] = f"Error: {str(e)}"
        else:
            responses[model_name] = "No response (API key not set)"
    return responses



# Synthesizer: utiliza o3-mini para resumir y comparar las respuestas
def synthesize_responses(responses, synthesizer_model="o3-mini", api_key=openai_api_key):
    prompt = f"You are a judge in a competition among {len(responses)} LLMs. Please summarize and compare the following responses from different LLMs:\n"
    for model, answer in responses.items():
        prompt += f"\nModel: {model}\nResponse: {answer}\n"
    prompt += "\nSummarize concisely and compare the responses. Your job is to evaluate each model's response and assign a score from 1 to 100. The model with the highest score is the best."
    
    messages = [{"role": "user", "content": prompt}]
    summary = get_llm_response(synthesizer_model, messages, api_key)

# Generate question using gpt-4o-mini
request = "Please come up with a challenging, nuanced question that I can ask a number of LLMs to evaluate their intelligence. Answer only with the question, no explanation."
messages = [{"role": "user", "content": request}]
question = get_llm_response("gpt-4o-mini", messages, openai_api_key)
print("Question:", question)

# Models to test
models = [
    ("gpt-4o-mini", openai_api_key, None),
    ("gemini-2.0-flash", google_api_key, "https://generativelanguage.googleapis.com/v1beta/openai/"),
    ("llama-3.3-70b-versatile", groq_api_key, "https://api.groq.com/openai/v1"),
    ("deepseek-chat", deepseek_api_key, "https://api.deepseek.com/v1")
   # ("llama3.2", "gemma", "http://localhost:11434/v1")
]

# Orquestar las respuestas de los distintos LLMs
llm_responses = coordinator_llm_responses(question, models)

# Mostrar cada respuesta individual
for model_name, answer in llm_responses.items():
    display(Markdown(f"### {model_name}\n{answer}"))

# Sintetizar las respuestas utilizando o3-mini para resumir y compararlas
synthesized_summary = synthesize_responses(llm_responses)

display(Markdown(f"""#Síntesis Comparativa {synthesized_summary}"""))