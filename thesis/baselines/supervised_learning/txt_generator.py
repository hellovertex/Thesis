import os
import sys

import numpy as np
import pandas as pd
from azureml.core import Experiment
from azureml.core import Workspace

from core.encoder import Encoder
from core.generator import Generator
from core.parser import Parser
from contextlib import contextmanager


class CsvGenerator(Generator):
    """This handles creation and population of folders inside train_data, corresponding to encoded
    PokerEpisode instances. These encodings can be used for supervised learning. """

    def __init__(self, data_dir: str,
                 out_dir,
                 parser: Parser,
                 encoder: Encoder,
                 out_filename: str,
                 write_azure: bool,
                 logfile="log.txt"):
        self._out_filename = out_filename
        self._out_dir = out_dir
        self._data_dir = data_dir
        self._parser = parser
        self._encoder = encoder
        self._write_azure = write_azure
        self._experiment = None
        self._logfile = logfile
        self._n_files_written_this_run = 0
        self._num_lines_written = 0

        with open(self._data_dir + logfile, "r") as f:
            self._n_files_already_encoded = len(f.readlines())
            print(f'reinitializing with {self._n_files_already_encoded} files already encoded')

        if write_azure:
            self._experiment = Experiment(workspace=self.get_workspace(),
                                          name="supervised-baseline")

    def __enter__(self):
        if self._write_azure:
            self._run = self._experiment.start_logging()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # print(exc_type, exc_value, exc_traceback)
        if self._write_azure:
            self._run.complete()

    @staticmethod
    def get_workspace():
        config = {
            "subscription_id": "0d263aec-21a1-4c68-90f8-687d99ccb93b",
            "resource_group": "thesis",
            "workspace_name": "generate-train-data"
        }
        # connect to get-train-data workspace
        return Workspace.get(name=config["workspace_name"],
                             subscription_id=config["subscription_id"],
                             resource_group=config["resource_group"])

    @property
    def out_filename(self):
        return self._out_filename

    def _write_metadata(self, file_dir):
        file_path_metadata = os.path.join(file_dir, f"{self._out_filename}.meta")
        with open(file_path_metadata, "a") as file:
            file.write(self._parser.metadata.__repr__()+"\n")
        return file_path_metadata

    def _write_train_data(self, data, labels, out_subdir):
        file_dir = os.path.join(self._out_dir, out_subdir)
        # create new file every 100k lines
        file_name = self._out_filename + '_' + str(int(self._num_lines_written / 100000))
        file_path = os.path.join(file_dir, file_name)
        if not os.path.exists(file_path):
            os.makedirs(os.path.realpath(file_dir), exist_ok=True)
        pd.DataFrame(data=data,
                     index=labels,
                     columns=self._encoder.feature_names).to_csv(
            file_path, index_label='label', mode='a')
        return file_dir, file_path

    def _write_to_azure(self, abs_filepath):
        self._run.upload_file(name="output.csv", path_or_stream=abs_filepath)

    def _log_progress(self, abs_filepath):
        with open(self._data_dir + self._logfile, "a") as f:
            f.write(abs_filepath + "\n")
        self._n_files_written_this_run += 1

    def generate_from_file(self, abs_filepath, out_subdir='0.25_0.50'):
        """Docstring"""
        parsed_hands = self._parser.parse_file(abs_filepath)
        training_data, labels = None, None
        i = 0  # in case parsed_hands is None
        for i, hand in enumerate(parsed_hands):
            observations, actions = self._encoder.encode_episode(hand)
            if not observations:
                continue
            if training_data is None:
                training_data = observations
                labels = actions
            else:
                try:
                    training_data = np.concatenate((training_data, observations), axis=0)
                    labels = np.concatenate((labels, actions), axis=0)
                except Exception as e:
                    print(e)
            self._num_lines_written += len(observations)
            print("Simulating environment", end='') if i == 0 else print('.', end='')

        # some rare cases, where the file did not contain showdown plays
        if training_data is None:
            return None

        print(f"\nExtracted {len(training_data)} training samples from {i + 1} poker hands"
              f"in file {self._n_files_written_this_run + self._n_files_already_encoded} {abs_filepath}...")

        self._log_progress(abs_filepath)
        # write train data
        file_dir, file_path = self._write_train_data(training_data, labels, out_subdir=out_subdir)

        # write meta data
        file_path_metadata = self._write_metadata(file_dir=file_dir)

        # write to cloud
        if self._write_azure:
            self._write_to_azure(file_path)

        # df = pd.read_csv(file_path)
        # print(f"Data created: and written to {file_path}, "
        #       f"metadata information is found at {file_path_metadata}")
        # print(df.head())
