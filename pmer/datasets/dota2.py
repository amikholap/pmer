from .base import BaseDataset


PLAYER_NAMES = {
    70388657: 'Dendi',
    86745912: 'Arteezy',
    87278757: 'Puppey',
    111620041: 'Sumail',
}


class Dota2Dataset(BaseDataset):

    player_names = PLAYER_NAMES
