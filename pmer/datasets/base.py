import ast
import datetime
import csv
import operator
import os

from .. import conf
from .. import Event


class BaseDataset(object):

    _date_format = '%Y-%m-%d %H:%M:%S'

    # A mapping of player ids to player names.
    # May be specified in subclasses.
    player_names = {}

    @classmethod
    def from_csv(cls, filename):
        """
        Load events from a CSV file.

        File should contain columns for date, winners and losers.
        """
        events = []
        path = os.path.join(conf.DATASET_DIR, filename)
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                date = datetime.datetime.strptime(row['date'], cls._date_format)
                winners = ast.literal_eval(row['winners'])
                losers = ast.literal_eval(row['losers'])
                event = Event(winners=winners, losers=losers, date=date)
                events.append(event)
        return cls(events)

    def __init__(self, events):
        """Construct a dataset from randomly ordered events."""
        events = sorted(events, key=operator.attrgetter('date'))
        self.events = events

    def __getitem__(self, key):
        return self.events[key]

    def __iter__(self):
        return iter(self.events)

    def __len__(self):
        return len(self.events)
