# importing Flask and other modules
import json
import os
import logging
import requests
from flask import Flask, request, render_template, jsonify

# Flask constructor
app = Flask(__name__)



# A decorator used to tell the application
# which URL is associated function
@app.route('/checkchurn', methods=["GET", "POST"])
def check_churn():
    if request.method == "GET":
        return render_template("input_form_page.html")

    elif request.method == "POST":
        prediction_input = [
            {
        "CreditScore": int(request.form.get("CreditScore")),  # getting input with name = ntp in HTML form
        "Geography": int(request.form.get("Geography")),  # getting input with name = pgc in HTML form
        "Gender": int(request.form.get("Gender")),
        "Age": int(request.form.get("Age")),
        "Tenure": int(request.form.get("Tenure")),
        "Balance": float(request.form.get("Balance")),
        "NumOfProducts": float(request.form.get("NumOfProducts")),
        "HasCrCard": int(request.form.get("HasCrCard")),
        "IsActiveMember": int(request.form.get("IsActiveMember")),
        "EstimatedSalary": int(request.form.get("EstimatedSalary"))
            }
                            ]

        logging.debug("Prediction input : %s", prediction_input)

        # use an environment variable to find the value of the diabetes prediction API
        predictor_api_url = os.environ['PREDICTOR_API']

        # use requests library to execute the prediction service API by sending an HTTP POST request
        res = requests.post(predictor_api_url, json=json.loads(json.dumps(prediction_input)))


        # json.dumps() function will convert a subset of Python objects into a json string.
        # json.loads() method can be used to parse a valid JSON string and convert it into a Python Dictionary.

        prediction_value = res.json()['result']
        logging.info("Prediction Output : %s", prediction_value)
        return render_template("response_page.html",
                               prediction_variable=prediction_value)


    else:

        return jsonify(message="Method Not Allowed"), 405  # The 405 Method Not Allowed should be used to indicate
        # that our app that does not allow the users to perform any other HTTP method (e.g., PUT and  DELETE) for
        # /checkchurn


# The code within this conditional block will only run the python file is executed as a
# script. See https://realpython.com/if-name-main-python/
if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 5000)), host='0.0.0.0', debug=True)
