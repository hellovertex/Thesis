from thesis.core.parser import PokerEpisode


class Encoder:
    """ Abstract Encoder Interface. All encoders should be derived from this base class
    and implement the method "encode_episode"."""

    def encode_episode(self, episode: PokerEpisode):
        """Encodes one PokerEpisode to a vector that can be used for machine learning."""
        raise NotImplementedError

