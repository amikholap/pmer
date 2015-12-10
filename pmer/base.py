import collections
import itertools
import numbers
import operator

import bintrees
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # pylint: disable=unused-import
import scipy.stats     # pylint: disable=unused-import


class Event(object):

    def __init__(self, winners, losers, date=None, weight=1):
        self.winners = winners
        self.losers = losers
        self.date = date
        self.weight = weight


class Rating(object):

    def __init__(self, value):
        self.value = value

    @property
    def params(self):
        return {
            'value': self.value,
        }

    def __float__(self):
        return self.value


class PlayerHistory(object):
    """Data structure to store player performance history."""

    class HistoricalRating(object):

        def __init__(self, rating, event):
            self.rating = rating
            self.event = event

    def __init__(self):
        self._tree = bintrees.AVLTree()

    def __iter__(self):
        return iter(self._tree.values())

    def __getitem__(self, key):
        """Get the most recent historical rating or a slice of them."""

        if isinstance(key, slice):
            return list(self._tree[key].values())

        # bintrees tree interface doesn't allow to get an element
        #     with key strictly less than asked.
        # So do it in two steps:
        #     1. Try to return previous item. Works if date is already there.
        #     2. Return any item with key <= date if it's not there.
        try:
            # Raises KeyError if item not found.
            hr = self._tree.prev_item(key)[1]
        except KeyError:
            try:
                # Raises KeyError if there are no keys less than date.
                hr = self._tree.floor_item(key)[1]
            except KeyError:
                hr = None

        return hr

    def __len__(self):
        return len(self._tree)

    def add(self, rating, event):
        """
        Save a record.

        Args:
            rating: Rating instance.
            event: Event that caused that rating.
        """
        hr = self.HistoricalRating(rating, event)
        self._tree[event.date] = hr


class RaterVisualisationMixin(object):
    """A container for rater-related visualisation methods."""

    def plot_rating_distribution_kde(self):
        means = [float(r) for r in self._ratings.values()]
        min_rating = min(means)
        max_rating = max(means)
        xs = np.linspace(min_rating - 0.1*abs(min_rating), max_rating + 0.1*abs(max_rating), 1000)
        ys = scipy.stats.gaussian_kde(means).pdf(xs)
        plt.plot(xs, ys)

    def plot_rating_history(self, player_ids):
        """
        Draw player rating history as time series.
        """

        # Allow both one and many player ids.
        if isinstance(player_ids, numbers.Number):
            player_ids = [player_ids]

        for player_id in player_ids:
            dates = []
            ratings = []
            player_history = self.history[player_id]
            for hr in player_history:
                dates.append(hr.event.date)
                ratings.append(hr.rating)
            label = self.player_names.get(player_id, str(player_id))
            self._plot_player_rating_history(dates, ratings, label=label)
        plt.legend()

    @staticmethod
    def _plot_player_rating_history(dates, ratings, label=None):
        plt.plot(dates, [float(r) for r in ratings], label=label)



class Rater(RaterVisualisationMixin):

    _rating_class = Rating

    def __init__(self, *, initial_rating_value=1):
        self._initial_rating_value = initial_rating_value
        self._ratings = collections.defaultdict(self._init_rating)
        self._history = collections.defaultdict(PlayerHistory)
        self.player_names = {}

    @property
    def history(self):
        return self._history

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

    def get_win_probabilities(self, team_a, team_b, date=None):
        return self._make_win_probabilities(team_a, team_b, date, predict=False)

    def predict_win_probabilities(self, team_a, team_b, date=None):
        return self._make_win_probabilities(team_a, team_b, date, predict=True)

    def _make_win_probabilities(self, team_a, team_b, date, predict):
        assert len(team_a) == len(team_b)
        if predict:
            team_a_ratings = self._predict_team_ratings(team_a, date=date)
            team_b_ratings = self._predict_team_ratings(team_b, date=date)
        else:
            team_a_ratings = self._get_team_ratings(team_a, date=date)
            team_b_ratings = self._get_team_ratings(team_b, date=date)
        team_a_win_p, team_b_win_p = \
            self._get_win_probabilities_for_ratings(team_a_ratings, team_b_ratings)
        return team_a_win_p, team_b_win_p

    def _get_team_ratings(self, team, date=None):
        raise NotImplementedError

    def _predict_team_ratings(self, team, date=None):
        return self._get_team_ratings(team, date=date)

    def _get_win_probabilities_for_ratings(self, team_a_ratings, team_b_ratings):
        raise NotImplementedError

    def make_leaderboard(self):
        """Return a sorted list of (player_id, rating) pairs."""
        leaderboard = [(player_id, float(rating)) for player_id, rating in self._ratings.items()]
        leaderboard = sorted(leaderboard, key=operator.itemgetter(1), reverse=True)
        return leaderboard

    def update_ratings(self, event):
        self._do_update_ratings(event)
        self._record_ratings_update(event)

    def _do_update_ratings(self, event):
        raise NotImplementedError

    def _record_ratings_update(self, event):
        """
        Record new ratings obtained from an event.
        """
        for player_id in itertools.chain(event.winners, event.losers):
            self.history[player_id].add(rating=self[player_id], event=event)

    def predict(self, events):
        """Calculate estimates for actual winners to win."""
        predictions = []
        for event in events:
            winners_pwin, _ = self.predict_win_probabilities(event.winners, event.losers, date=event.date)
            predictions.append(winners_pwin)
        return predictions

    def process_dataset(self, dataset):
        """Calculate ratings that result from the provided dataset."""
        for event in dataset:
            self.update_ratings(event)
        self.player_names = dataset.player_names
