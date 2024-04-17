from flask import Flask, jsonify
import helper

app = Flask(__name__)

@app.route('/QnA/<string:question>', methods=['GET'])
def get_information(question):
    data = helper.get_information(question)
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True)