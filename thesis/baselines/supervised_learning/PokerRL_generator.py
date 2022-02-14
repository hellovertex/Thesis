from core.parser import Parser
from core.encoder import Encoder
import os
import pandas as pd
import numpy as np


class PokerRLGenerator:
    """This handles creation of folders inside train_data that correspond to table specifics.
     Table specifics include number of players, platform, blinds, table-type, poker-variant,..."""

    # todo consider writing base class
    def __init__(self, data_dir: str,
                 out_dir,
                 parser: Parser,
                 encoder: Encoder,
                 out_filename: str):
        self._out_filename = out_filename
        self._out_dir = out_dir
        self._data_dir = data_dir
        self._parser = parser
        self._encoder = encoder

    @property
    def out_filename(self):
        return self._out_filename

    def _write_metadata(self, file_dir):
        file_path_metadata = os.path.join(file_dir, f"{self._out_filename}.meta")
        with open(file_path_metadata, "w") as file:
            file.write(self._parser.metadata.__repr__())
        return file_path_metadata

    def _write_train_data(self, data, labels, out_subdir):
        file_dir = os.path.join(self._out_dir, out_subdir)
        file_path = os.path.join(file_dir, self._out_filename)
        if not os.path.exists(file_path):
            os.makedirs(os.path.realpath(file_dir))
        pd.DataFrame(data=data,
                     index=labels,
                     columns=self._encoder.feature_names).to_csv(
            file_path, index_label='label', mode='a')
        return file_dir, file_path

    def generate_from_file(self, abs_filepath, out_subdir='0.25_0.50'):
        """Docstring"""
        parsed_hands = self._parser.parse_file(abs_filepath)
        training_data, labels = None, None
        for i, hand in enumerate(parsed_hands):
            observations, actions = self._encoder.encode_episode(hand)
            if training_data is None:
                training_data = observations
                labels = actions
            else:
                training_data = np.concatenate((training_data, observations), axis=0)
                labels = np.concatenate((labels, actions), axis=0)
            print("Simulating environment", end='') if i == 0 else print('.', end='')

        print(f"\nExtracted {len(training_data)} training samples from {i + 1} poker hands...")

        # write train data
        file_dir, file_path = self._write_train_data(training_data, labels, out_subdir=out_subdir)

        # write meta data
        file_path_metadata = self._write_metadata(file_dir=file_dir)

        df = pd.read_csv(file_path)
        print(f"Data created: and written to {file_path}, "
              f"metadata information is found at {file_path_metadata}")
        print(df.head())
