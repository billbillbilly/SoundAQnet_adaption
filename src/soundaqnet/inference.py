"""
soundaqnet.inference
====================
SoundAQnet model loading and inference pipeline.

CLI
---
    soundaqnet-infer \\
        --dataset      <Dataset_dir>      \\
        --dataset_mel  <mel_npy_dir>      \\
        --dataset_wav_loudness <loud_dir> \\
        --model        <checkpoint.pth>   \\
        --event_output_dir  SoundAQnet_event_probability  \\
        --paq_output_dir    SoundAQnet_scene_ISOPl_ISOEv_PAQ8DAQs
"""

from __future__ import annotations

import os
import sys
import argparse

import torch
import numpy as np
from sklearn import metrics

from soundaqnet.framework.processing import forward_for_LLM
from soundaqnet.framework.data_generator import DataGenerator_Mel_loudness_graph
from soundaqnet.framework.models_pytorch import SoundAQnet
from soundaqnet.framework.pytorch_utils import count_parameters
from soundaqnet.framework import config
from soundaqnet.framework.utilities import create_folder


def cal_auc(targets_event: np.ndarray, outputs_event: np.ndarray) -> float:
    aucs = []
    for i in range(targets_event.shape[0]):
        test_y_auc, pred_auc = targets_event[i, :], outputs_event[i, :]
        if np.sum(test_y_auc):
            test_auc = metrics.roc_auc_score(test_y_auc, pred_auc)
            aucs.append(test_auc)
    return sum(aucs) / len(aucs)


def main() -> int:
    """CLI: soundaqnet-infer"""
    parser = argparse.ArgumentParser(
        description="Run SoundAQnet inference on pre-extracted features.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=os.path.join(os.getcwd(), "Dataset"),
        help="Path to the Dataset directory (for normalization files).",
    )
    parser.add_argument(
        "--dataset_mel",
        type=str,
        required=True,
        help="Path to the mel feature directory (.npy files).",
    )
    parser.add_argument(
        "--dataset_wav_loudness",
        type=str,
        required=True,
        help="Path to the wav loudness feature directory (.npy files).",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model .pth checkpoint file.",
    )
    parser.add_argument(
        "--event_output_dir",
        type=str,
        default=os.path.join(os.getcwd(), "SoundAQnet_event_probability"),
        help="Directory to save event probability outputs.",
    )
    parser.add_argument(
        "--paq_output_dir",
        type=str,
        default=os.path.join(os.getcwd(), "SoundAQnet_scene_ISOPl_ISOEv_PAQ8DAQs"),
        help="Directory to save scene / ISO / PAQ outputs.",
    )
    args = parser.parse_args()

    node_emb_dim = 64
    hidden_dim, out_dim = 32, 64
    batch_size = 32
    config.batch_size = batch_size
    number_of_nodes = 8

    model = SoundAQnet(
        max_node_num=number_of_nodes,
        node_emb_dim=node_emb_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
    )

    checkpoint = torch.load(args.model, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    if config.cuda:
        model.cuda()

    dataset_path = args.dataset
    dataset_mel = args.dataset_mel
    dataset_loudness = args.dataset_wav_loudness

    generator = DataGenerator_Mel_loudness_graph(
        dataset_path, node_emb_dim, number_of_nodes
    )

    generate_func = generator.generate_inference_soundscape_clip_for_LLM(
        dataset_mel, dataset_loudness
    )
    output_dict = forward_for_LLM(
        model=model, generate_func=generate_func, cuda=config.cuda
    )

    result_event_dir = args.event_output_dir
    create_folder(result_event_dir)

    result_PAQ_dir = args.paq_output_dir
    create_folder(result_PAQ_dir)

    for each_index, name in enumerate(output_dict["audio_names"]):
        print("Soundscape audio clip:", name.split(".npy")[0])

        txtfile = os.path.join(result_event_dir, name.replace(".npy", "_event.txt"))
        np.savetxt(txtfile, output_dict["output_event"][each_index])
        print("Audio event probability matrix:", output_dict["output_event"][each_index])

        txtfile = os.path.join(result_PAQ_dir, name.replace(".npy", "_scene_PAQ.txt"))

        with open(txtfile, "w") as f:
            max_id = np.argmax(output_dict["output_scene"][each_index])
            print("Acoustic scene:", config.scene_labels[max_id])
            f.write(config.scene_labels[max_id] + "\n")

            isop = str(output_dict["output_ISOPls"][each_index][0])
            isoe = str(output_dict["output_ISOEvs"][each_index][0])
            f.write(isop + "\t" + isoe + "\n")
            print("ISOP and ISOE:", isop + "\t" + isoe)

            paq_vals = [
                str(output_dict["output_pleasant"][each_index][0]),
                str(output_dict["output_eventful"][each_index][0]),
                str(output_dict["output_chaotic"][each_index][0]),
                str(output_dict["output_vibrant"][each_index][0]),
                str(output_dict["output_uneventful"][each_index][0]),
                str(output_dict["output_calm"][each_index][0]),
                str(output_dict["output_annoying"][each_index][0]),
                str(output_dict["output_monotonous"][each_index][0]),
            ]
            paq_line = "\t".join(paq_vals)
            print("PAQ 8D AQs:", paq_line)
            f.write(paq_line + "\n")

        print()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, IOError) as e:
        sys.exit(e)
