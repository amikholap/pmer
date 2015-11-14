import ast
import csv
import datetime
import math

import numpy as np

from .base import Event


def events_from_csv(filename):
    date_fmt = '%Y-%m-%d %H:%M:%S'
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            date = datetime.datetime.strptime(row['date'], date_fmt)
            winners = ast.literal_eval(row['winners'])
            losers = ast.literal_eval(row['losers'])
            event = Event(winners, losers, date=date)
            yield event


def logloss_for_dataset(raters, filename):
    events = events_from_csv(filename)
    errors = np.zeros(len(raters))
    for event in events:
        for i, rater in enumerate(raters):
            winners_pwin, losers_pwin = rater.get_win_probabilities(event.winners, event.losers)
            event_error = - math.log(winners_pwin)
            # event_error = (1 - winners_pwin) ** 2
            errors[i] += event_error
            rater.update_ratings(event)
    return errors
