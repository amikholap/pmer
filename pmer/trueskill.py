import datetime
import math
import operator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trueskill

from .base import Rater, Rating, RaterVisualisationMixin


class TrueskillRating(Rating):

    @property
    def params(self):
        return {
            'mu': self._rating.mu,
            'sigma': self._rating.sigma,
        }

    # pylint: disable=super-init-not-called
    def __init__(self, rating):
        """Wrap trueskill.Rating object."""
        self._rating = rating

    def __float__(self):
        return float(self._rating)

    def __getattr__(self, name):
        return getattr(self._rating, name)


class TrueskillRaterVisualisationMixin(RaterVisualisationMixin):

    @staticmethod
    def _plot_player_rating_history(dates, ratings, label=None):
        lower_bounds = []
        means = []
        upper_bounds = []
        for r in ratings:
            lower_bounds.append(r.mu - 3*r.sigma)
            means.append(r.mu)
            upper_bounds.append(r.mu + 3*r.sigma)
        line, = plt.plot(dates, means, label=label)
        plt.fill_between(dates, upper_bounds, lower_bounds, color=line.get_c(), alpha=0.2)


class TrueskillRater(TrueskillRaterVisualisationMixin, Rater):

    # Rating class is not used for explicit object construction.
    # Ratings are created by a wrapper function.
    _rating_class = TrueskillRating

    def __init__(self, *, mu=25.0, sigma=25/3, beta=25/6, tau=25/300):
        super().__init__(initial_rating_value=mu)
        self._env = trueskill.TrueSkill(mu=mu, sigma=sigma, beta=beta, tau=tau, draw_probability=0.0, backend='scipy')

    def _init_rating(self):
        return self.create_rating()

    def create_rating(self, *args, **kwargs):
        return TrueskillRating(self._env.create_rating(*args, **kwargs))

    def _get_team_ratings(self, team, date=None):
        if date is None:
            team_ratings = [self[player_id] for player_id in team]
        else:
            team_ratings = [self.history[player_id][date].rating for player_id in team]
        return team_ratings

    def _get_win_probabilities_for_ratings(self, team_a_ratings, team_b_ratings):
        delta_mu = sum([x.mu for x in team_a_ratings]) - sum([x.mu for x in team_b_ratings])
        sum_sigma = sum([x.sigma ** 2 for x in team_a_ratings]) + sum([x.sigma ** 2 for x in team_b_ratings])
        playerCount = len(team_a_ratings) + len(team_b_ratings)
        denominator = math.sqrt(playerCount * (self._env.beta * self._env.beta) + sum_sigma)

        team_a_win_probability = self._env.cdf(delta_mu / denominator)
        team_b_win_probability = 1 - team_a_win_probability

        return team_a_win_probability, team_b_win_probability

    def make_leaderboard(self):
        """
        Return a sorted list of (player_id, rating) pairs.

        Trueskill leaderboard is based on conservative player ratings (mu - 3*sigma).
        """
        leaderboard = [(player_id, rating.mu - 3*rating.sigma) for player_id, rating in self._ratings.items()]
        leaderboard = sorted(leaderboard, key=operator.itemgetter(1), reverse=True)
        return leaderboard

    def _do_update_ratings(self, event):
        assert len(event.winners) == len(event.losers)

        winner_ratings = self._get_team_ratings(event.winners)
        loser_ratings = self._get_team_ratings(event.losers)

        new_winner_ratings, new_loser_ratings = self._env.rate([winner_ratings, loser_ratings], ranks=[0, 1])

        for i, player_id in enumerate(event.winners):
            self[player_id] = TrueskillRating(new_winner_ratings[i])
        for i, player_id in enumerate(event.losers):
            self[player_id] = TrueskillRating(new_loser_ratings[i])


class ExponentiallySmoothedTrueskillRater(TrueskillRater):

    def _predict_team_ratings(self, team, date=None):
        if date is None:
            date = datetime.datetime.max
        player_histories = {player_id: self.history[player_id][:date] for player_id in team}
        smoothed_player_ratings = []
        for player_id, ph in player_histories.items():
            historical_ratings = np.array([hr.rating.mu for hr in ph])
            smoothed_ratings = pd.ewma(historical_ratings, span=5)
            if len(smoothed_ratings) > 0:
                params = ph[-1].rating.params
                params['mu'] = smoothed_ratings[-1]
            else:
                params = self[player_id].params
                params['mu'] = self[player_id].mu
            smoothed_player_ratings.append(self.create_rating(**params))
        return smoothed_player_ratings
