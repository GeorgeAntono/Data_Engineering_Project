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
        CreditScore = int(request.form.get("CreditScore"))  # getting input with name = ntp in HTML form
        Geography = int(request.form.get("Geography"))  # getting input with name = pgc in HTML form
        Gender = int(request.form.get("Gender"))
        Age = int(request.form.get("Age"))
        Tenure = int(request.form.get("Tenure"))
        Balance = float(request.form.get("Balance"))
        NumOfProducts = float(request.form.get("NumOfProducts"))
        HasCrCard = int(request.form.get("HasCrCard"))
        IsActiveMember = int(request.form.get("IsActiveMember"))
        EstimatedSalary = int(request.form.get("EstimatedSalary"))

        # we will replace this simple (and inaccurate logic) with a prediction from a machine learning model in a
        # future in a future lab
        if EstimatedSalary > 12000:
            prediction_value = True
        else:
            prediction_value = False

        return render_template("response_page.html",
                               prediction_variable=prediction_value)

    else:
        return jsonify(message="Method Not Allowed"), 405  # The 405 Method Not Allowed should be used to indicate
    # that our app that does not allow the users to perform any other HTTP method (e.g., PUT and  DELETE) for
    # '/checkdiabetes' path


# The code within this conditional block will only run the python file is executed as a
# script. See https://realpython.com/if-name-main-python/
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
