import datetime
import math

import numpy as np
import pandas as pd

from .base import Rater, Rating


class EloRating(Rating):
    pass


class EloRater(Rater):

    _rating_class = EloRating

    def __init__(self, *, K=0.1, scale=0.5, **kwargs):
        super().__init__(**kwargs)
        self.K = K
        self.scale = scale

    def _get_team_ratings(self, team, date=None):
        if date is None:
            team_rating_sum = sum([self[player_id].value for player_id in team])
        else:
            team_rating_sum = sum([self.history[player_id][date].rating.value for player_id in team])
        return team_rating_sum

    def _get_win_probabilities_for_ratings(self, rating_a, rating_b):
        a_pwin = 1 / (1 + math.exp((rating_b - rating_a) / self.scale))
        b_pwin = 1 - a_pwin
        return a_pwin, b_pwin

    def _do_update_ratings(self, event):
        assert len(event.winners) == len(event.losers)

        winners_rating = self._get_team_ratings(event.winners)
        losers_rating = self._get_team_ratings(event.losers)

        winners_pwin, _ = self._get_win_probabilities_for_ratings(winners_rating, losers_rating)

        delta = event.weight * self.K * (1 - winners_pwin)

        # Update rating values.
        # Higher relative rating causes higher update.
        for player_id in event.winners:
            self[player_id] = self.create_rating(
                value=self[player_id].value + delta * (self[player_id].value / winners_rating)
            )
        for player_id in event.losers:
            self[player_id] = self.create_rating(
                value=self[player_id].value - delta * (self[player_id].value / losers_rating)
            )


class ExponentiallySmoothedEloRater(EloRater):

    def _predict_team_ratings(self, team, date=None):
        if date is None:
            date = datetime.datetime.max
        player_histories = {player_id: self.history[player_id][:date] for player_id in team}
        smoothed_player_ratings = []
        for player_id, ph in player_histories.items():
            historical_ratings = np.array([hr.rating.value for hr in ph])
            smoothed_ratings = pd.ewma(historical_ratings, span=10)
            if len(smoothed_ratings) > 0:
                smoothed_value = smoothed_ratings[-1]
            else:
                smoothed_value = self[player_id].value
            smoothed_player_ratings.append(smoothed_value)
        return sum(smoothed_player_ratings)
