from flask import Flask, render_template, url_for
from flask import request as req  
from transformers import AutoTokenizer, AutoModelWithLMHead
import torch

app = Flask(_name_)

model_name = "flax-community/t5-base-cnn-dm"
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_t5 = AutoModelWithLMHead.from_pretrained(model_name).to(device)

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/summarize', methods=["POST"])
def summarize():

    if req.method == "POST":

        inputtext = req.form["inputtext_"]  

        input_text = "summarize: " + inputtext

        tokenized_text = tokenizer.encode(input_text, return_tensors='pt', max_length=1024, truncation=True, padding="max_length").to(device)
        summary_ = model_t5.generate(tokenized_text, min_length=100, max_length=1000, num_beams=4, no_repeat_ngram_size=2) 
        summary = tokenizer.decode(summary_[0], skip_special_tokens=True)

    return render_template("output.html", data={"summary": summary})


if _name_ == '_main_':
    app.run(debug=True, port=8000)