import math

from .base import Rater, Rating


class EloRating(Rating):
    pass


class EloRater(Rater):

    _rating_class = EloRating

    def __init__(self, *, K=0.1, scale=0.5, **kwargs):
        super().__init__(**kwargs)
        self.K = K
        self.scale = scale

    def get_win_probabilities(self, team_a, team_b):
        assert len(team_a) == len(team_b)

        team_a_rating = self._get_team_rating(team_a)
        team_b_rating = self._get_team_rating(team_b)

        team_a_pwin, team_b_pwin = self._get_win_probabilities_for_ratings(team_a_rating, team_b_rating)

        return team_a_pwin, team_b_pwin

    def _get_team_rating(self, team):
        team_rating_sum = sum([self[player_id].value for player_id in team])
        return team_rating_sum

    def _get_win_probabilities_for_ratings(self, rating_a, rating_b):
        a_pwin = 1 / (1 + math.exp((rating_b - rating_a) / self.scale))
        b_pwin = 1 - a_pwin
        return a_pwin, b_pwin

    def _do_update_ratings(self, event):
        assert len(event.winners) == len(event.losers)

        winners_rating = self._get_team_rating(event.winners)
        losers_rating = self._get_team_rating(event.losers)

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
