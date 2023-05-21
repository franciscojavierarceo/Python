from flask import Flask, render_template, request
from flask import jsonify
import random
import datetime

#Flask initialization
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/hello", methods=["GET","POST"])
def hello():
    return jsonify('this is a test')


@app.route("/chat", methods=["GET","POST"])
def chatbot_response():
    print(request.data)
    messages = [
        'this is an example response',
        'this is another example response',
        'this is another response',
        'this too is a response',
        'blah blah blah',

    ]
    random.seed(datetime.datetime.now().timestamp())
    #msg = request.form["msg"]
    #response = chatbot.get_response(msg)
    

    return jsonify({"msg": random.choice(messages)})

if __name__ == "__main__":
 app.run(debug=True)
