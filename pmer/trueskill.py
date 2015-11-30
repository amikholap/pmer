import math
import operator

import matplotlib.pyplot as plt
import trueskill

from .base import Rater, RaterVisualisationMixin


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

    # Explicit rating class is not required.
    # Ratings are created using a wrapper function.
    _rating_class = trueskill.Rating

    def __init__(self, *, mu=25.0, sigma=25/3, beta=25/6, tau=25/300):
        super().__init__(initial_rating_value=mu)
        self._env = trueskill.TrueSkill(mu=mu, sigma=sigma, beta=beta, tau=tau, draw_probability=0.0, backend='scipy')

    def _init_rating(self):
        return self.create_rating()

    def _get_player_ratings(self, players):
        return [self[player] for player in players]

    def create_rating(self, *args, **kwargs):
        return self._env.create_rating(*args, **kwargs)

    def get_win_probabilities(self, team_a, team_b):
        assert len(team_a) == len(team_b)

        team_a_ratings = self._get_player_ratings(team_a)
        team_b_ratings = self._get_player_ratings(team_b)

        delta_mu = sum([x.mu for x in team_a_ratings]) - sum([x.mu for x in team_b_ratings])
        sum_sigma = sum([x.sigma ** 2 for x in team_a_ratings]) + sum([x.sigma ** 2 for x in team_b_ratings])
        playerCount = len(team_a) + len(team_b)
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

        winner_ratings = self._get_player_ratings(event.winners)
        loser_ratings = self._get_player_ratings(event.losers)

        new_winner_ratings, new_loser_ratings = self._env.rate([winner_ratings, loser_ratings], ranks=[0, 1])

        for i, player_id in enumerate(event.winners):
            self[player_id] = new_winner_ratings[i]
        for i, player_id in enumerate(event.losers):
            self[player_id] = new_loser_ratings[i]
