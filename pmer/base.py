import collections


class Event(object):

    def __init__(self, winners, losers, date=None):
        self.winners = winners
        self.losers = losers
        self.date = date


class Rating(object):

    def __init__(self, value):
        self.value = value


class Rater(object):

    _rating_class = Rating

    def __init__(self, *, initial_rating_value=1):
        self._initial_rating_value = initial_rating_value
        self._ratings = collections.defaultdict(self._init_rating)

    def __getitem__(self, key):
        return self._ratings[key]

    def __setitem__(self, key, value):
        self._ratings[key] = value

    def _init_rating(self):
        initial_params = self._get_initial_rating_params()
        rating = self._rating_class(**initial_params)
        return rating

    def _get_initial_rating_params(self):
        params = {
            'value': self._initial_rating_value,
        }
        return params

    def create_rating(self, *args, **kwargs):
        return self._rating_class(*args, **kwargs)

    def get_win_probabilities(self, team_a, team_b):
        raise NotImplementedError

    def update_ratings(self, event):
        raise NotImplementedError
