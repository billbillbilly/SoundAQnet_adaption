import os
import pickle
import time

import numpy as np

from soundaqnet.framework import config
from soundaqnet.framework.utilities import calculate_scalar, create_folder, scale


class DataGenerator_Mel_loudness_graph(object):
    def __init__(
        self,
        Dataset_path,
        node_emb_dim,
        number_of_nodes=8,
        seed=42,
        normalization=True,
        overwrite=False,
    ):
        self.Dataset_path = Dataset_path
        self.batch_size = config.batch_size
        self.random_state = np.random.RandomState(seed)
        self.validate_random_state = np.random.RandomState(0)
        self.test_random_state = np.random.RandomState(0)

        # Load data
        load_time = time.time()

        # Graph topology is now handled inside GatedGCNLayer as fixed buffers.
        # No DGL graph object is needed at data-loading time.

        self.normal = normalization
        output_dir = os.path.join(Dataset_path, "0_normalization_files")
        create_folder(output_dir)
        normalization_log_mel_file = os.path.join(output_dir, "norm_log_mel.pickle")
        normalization_loudness_file = os.path.join(output_dir, "norm_loudness.pickle")

        # Fall back to bundled normalization files when local ones are absent.
        # This happens on the first inference run against a fresh Dataset_path,
        # or when users point --dataset at a new directory.
        if self.normal and (
            not os.path.exists(normalization_log_mel_file)
            or not os.path.exists(normalization_loudness_file)
        ):
            try:
                from importlib.resources import files as _pkg_files

                _data_pkg = _pkg_files("soundaqnet.data")
                _bml = str(_data_pkg / "norm_log_mel.pickle")
                _bln = str(_data_pkg / "norm_loudness.pickle")
                if os.path.isfile(_bml) and os.path.isfile(_bln):
                    normalization_log_mel_file = _bml
                    normalization_loudness_file = _bln
                    print("[soundaqnet] Using bundled normalization files.")
                else:
                    print("[soundaqnet] Warning: bundled normalization files not found at", _bml)
            except Exception as _exc:
                print(f"[soundaqnet] Warning: could not locate bundled normalization files: {_exc}")

        if self.normal and not os.path.exists(normalization_loudness_file) or overwrite:
            norm_pickle = {}
            self.mean_log_mel, self.std_log_mel = calculate_scalar(np.concatenate(self.train_x))
            norm_pickle["mean"] = self.mean_log_mel
            norm_pickle["std"] = self.std_log_mel
            self.save_pickle(norm_pickle, normalization_log_mel_file)

            norm_pickle = {}
            self.mean_loudness, self.std_loudness = calculate_scalar(
                np.concatenate(self.train_x_loudness)
            )
            norm_pickle["mean"] = self.mean_loudness
            norm_pickle["std"] = self.std_loudness
            self.save_pickle(norm_pickle, normalization_loudness_file)
        else:
            print("using: ", normalization_log_mel_file)
            norm_pickle = self.load_pickle(normalization_log_mel_file)
            self.mean_log_mel = norm_pickle["mean"]
            self.std_log_mel = norm_pickle["std"]
            print("Log Mel Mean: ", self.mean_log_mel)
            print("Log Mel STD: ", self.std_log_mel)

            print("using: ", normalization_loudness_file)
            norm_pickle = self.load_pickle(normalization_loudness_file)
            self.mean_loudness = norm_pickle["mean"]
            self.std_loudness = norm_pickle["std"]
            print("Loudness Mean: ", self.mean_loudness)
            print("Loudness STD: ", self.std_loudness)

        print("norm: ", self.mean_log_mel.shape, self.std_log_mel.shape)
        print("norm: ", self.mean_loudness.shape, self.std_loudness.shape)

        print("Loading data time: {:.3f} s".format(time.time() - load_time))

    def get_input_output(self, all_data):
        ISOPls, ISOEvs = self.get_ISOPl_ISOEv(all_data)
        scene_labels = self.load_scene_labels(all_data)

        (
            audio_names,
            features,
            _,
            pleasant,
            eventful,
            chaotic,
            vibrant,
            uneventful,
            calm,
            annoying,
            monotonous,
        ) = (
            all_data["soundscape"],
            all_data["feature_names"],
            all_data["masker"],
            all_data["pleasant"],
            all_data["eventful"],
            all_data["chaotic"],
            all_data["vibrant"],
            all_data["uneventful"],
            all_data["calm"],
            all_data["annoying"],
            all_data["monotonous"],
        )

        audio_names = all_data["feature_names"]

        assert all_data["all_events"] == config.event_labels
        sound_maskers_labels = all_data["event_labels"]

        event_labels = np.zeros((len(sound_maskers_labels), len(config.event_labels)))
        for i, each in enumerate(sound_maskers_labels):
            for sub_each in each:
                event_labels[i, config.event_labels.index(sub_each)] = 1

        pleasant, eventful, chaotic, vibrant, uneventful, calm, annoying, monotonous = (
            np.array(pleasant)[:, None],
            np.array(eventful)[:, None],
            np.array(chaotic)[:, None],
            np.array(vibrant)[:, None],
            np.array(uneventful)[:, None],
            np.array(calm)[:, None],
            np.array(annoying)[:, None],
            np.array(monotonous)[:, None],
        )

        return (
            features,
            scene_labels,
            event_labels,
            ISOPls,
            ISOEvs,
            pleasant,
            eventful,
            chaotic,
            vibrant,
            uneventful,
            calm,
            annoying,
            monotonous,
            np.array(audio_names),
        )

    def get_ISOPl_ISOEv(self, all_data):
        attributes = [
            "pleasant",
            "eventful",
            "chaotic",
            "vibrant",
            "uneventful",
            "calm",
            "annoying",
            "monotonous",
        ]
        ISOPl_weights = [
            1,
            0,
            -np.sqrt(2) / 2,
            np.sqrt(2) / 2,
            0,
            np.sqrt(2) / 2,
            -1,
            -np.sqrt(2) / 2,
        ]
        ISOEv_weights = [
            0,
            1,
            np.sqrt(2) / 2,
            np.sqrt(2) / 2,
            -1,
            -np.sqrt(2) / 2,
            0,
            -np.sqrt(2) / 2,
        ]

        emotion_values = [all_data[each] for each in attributes]

        emotion_values = np.array(emotion_values).transpose((1, 0))

        ISOPls = (emotion_values * ISOPl_weights).sum(axis=1) / (4 + np.sqrt(32))
        ISOEvs = (emotion_values * ISOEv_weights).sum(axis=1) / (4 + np.sqrt(32))
        ISOPls, ISOEvs = ISOPls[:, None], ISOEvs[:, None]
        return ISOPls, ISOEvs

    def load_scene_labels(self, all_data):
        USotW_acoustic_scene_laebls = all_data["USotW_acoustic_scene_labels"]
        clips = all_data["soundscape"]

        scenes = [USotW_acoustic_scene_laebls[each.split("_44100")[0]] for each in clips]
        correct_scene = []
        for each in scenes:
            if each == "park ":
                correct_scene.append("park")
            else:
                correct_scene.append(each)

        scene_labels = np.array([config.scene_labels.index(each) for each in correct_scene])

        return scene_labels

    def load_pickle(self, file):
        with open(file, "rb") as f:
            data = pickle.load(f)
        return data

    def save_pickle(self, data, file):
        with open(file, "wb") as f:
            pickle.dump(data, f)

    def generate_inference_soundscape_clip_for_LLM(self, Dataset_mel, Dataset_loudness):
        # Keep a stable order so mel and loudness match
        file_names = sorted([f for f in os.listdir(Dataset_mel) if f.endswith(".npy")])

        audios_num = len(file_names)

        pointer = 0
        while pointer < audios_num:
            batch_file_names = file_names[pointer : pointer + self.batch_size]
            pointer += self.batch_size

            batch_x_list = []
            batch_x_loudness_list = []

            for file_name in batch_file_names:
                mel_path = os.path.join(Dataset_mel, file_name)
                loudness_path = os.path.join(Dataset_loudness, file_name)

                if not os.path.exists(loudness_path):
                    raise FileNotFoundError(
                        f"Loudness feature file not found for {file_name}: {loudness_path}"
                    )

                mel = np.load(mel_path).astype(np.float32, copy=False)
                loudness = np.load(loudness_path).astype(np.float32, copy=False)

                batch_x_list.append(mel)
                batch_x_loudness_list.append(loudness)

            batch_x = np.stack(batch_x_list, axis=0)
            batch_x_loudness = np.stack(batch_x_loudness_list, axis=0)

            if self.normal:
                batch_x = self.transform(batch_x, self.mean_log_mel, self.std_log_mel)
                batch_x_loudness = self.transform(
                    batch_x_loudness, self.mean_loudness, self.std_loudness
                )

            yield batch_x, batch_x_loudness, batch_file_names

    def transform(self, x, mean, std):
        """Transform data.

        Args:
          x: (batch_x, seq_len, freq_bins) | (seq_len, freq_bins)

        Returns:
          Transformed data.
        """
        return scale(x, mean, std)
