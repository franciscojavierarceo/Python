from flask import Flask, render_template, request
from flask import jsonify
#from chatterbot import ChatBot
#from chatterbot.trainers import ChatterBotCorpusTrainer

#Flask initialization
app = Flask(__name__)
#chatbot=ChatBot('Pythonscholar')

# Create a new trainer for the chatbot
#trainer = ChatterBotCorpusTrainer(chatbot)
# Now let us train our bot with multiple corpus

#trainer.train("chatterbot.corpus.english.greetings","chatterbot.corpus.english.conversations")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/hello", methods=["GET","POST"])
def hello():
    return jsonify('this is a test')


@app.route("/chat", methods=["GET","POST"])
def chatbot_response():
    print(request.data)
    #msg = request.form["msg"]
    #response = chatbot.get_response(msg)
    return jsonify({"msg": "this is a test"})

if __name__ == "__main__":
 app.run(debug=True)
