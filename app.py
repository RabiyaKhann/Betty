from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)
CORS(app)

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Define a variable to store chat history
chat_history_ids = None

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        msg = request.form["msg"]
        response = get_Chat_response(msg)
        return jsonify({"response": response})
    else:
        return "Method not allowed", 405

def get_Chat_response(text):
    global chat_history_ids

    # Encode user input and generate response
    new_user_input_ids = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    return response

if __name__ == '__main__':
    app.run(port=int(os.environ.get('PORT', 5000)))
