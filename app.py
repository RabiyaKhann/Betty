from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Load model and tokenizer from Hugging Face's model hub
try:
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
except Exception as e:
    print(f"Failed to load model from Hugging Face: {e}")
    tokenizer = None
    model = None

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/get", methods=["POST"])
def chat():
    if model is None or tokenizer is None:
        return jsonify({"response": "Failed to load model from Hugging Face. Check logs for details."})

    try:
        msg = request.form["msg"]
        response = get_chat_response(msg)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"response": f"Error: {e}"})

def get_chat_response(text):
    new_user_input_ids = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors='pt')
    chat_history_ids = model.generate(new_user_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(chat_history_ids[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

if __name__ == '__main__':
    app.run(debug=True)
