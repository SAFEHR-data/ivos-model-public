import pandas as pd

from primitivo_model.data.criteria import evaluate_all_task_predictions
from primitivo_model.data.generator import TaskSet
from primitivo_model.naive.model import collect_predictions


def predict_meets_criteria(tasks: TaskSet):
    task_preds = collect_predictions(tasks)
    criteria = tasks.get_clinical_criteria(standardised=True)
    measurement_names = tasks.measurement_names
    return pd.Series(evaluate_all_task_predictions(task_preds, measurement_names, criteria))
