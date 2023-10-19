import os

import pandas as pd

import churn_predictor


class TestChurnPredictor:
    def test_predict_single_record(self):
        test_dir = os.path.dirname(os.path.abspath(__file__))
        test_data_file = os.path.join(test_dir, "../testResources/prediction_request.json")
        test_model_file = os.path.join(test_dir, "../testResources/model.pkl")
        with open(test_data_file) as json_file:
            data = pd.read_json(json_file)
        cp = churn_predictor.ChurnPredictor(model_file=test_model_file)
        status = cp.predict_single_record(data)
       
        assert bool(status[0]) is not None
        assert bool(status[0]) is False