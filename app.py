from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    # Example response for testing
    response = f"You said: {msg}"
    return response

if __name__ == "__main__":
    app.run(debug=True)
