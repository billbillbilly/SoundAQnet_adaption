import os
import sys
import argparse
import torch
import numpy as np

sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])
from framework.processing import *
from framework.data_generator import *
from framework.models_pytorch import *
from framework.pytorch_utils import count_parameters


def cal_auc(targets_event, outputs_event):
    aucs = []
    for i in range(targets_event.shape[0]):
        test_y_auc, pred_auc = targets_event[i, :], outputs_event[i, :]
        if np.sum(test_y_auc):
            test_auc = metrics.roc_auc_score(test_y_auc, pred_auc)
            aucs.append(test_auc)
    final_auc_event_branch = sum(aucs) / len(aucs)
    return final_auc_event_branch


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default=os.path.join(os.getcwd(), "Dataset"),
        help="Path to the Dataset directory used by DataGenerator_Mel_loudness_graph."
    )
    parser.add_argument(
        "--dataset_mel",
        type=str,
        required=True,
        help="Path to the mel feature directory."
    )
    parser.add_argument(
        "--dataset_wav_loudness",
        type=str,
        required=True,
        help="Path to the wav loudness feature directory."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model .pth file."
    )
    parser.add_argument(
        "--event_output_dir",
        type=str,
        default=os.path.join(os.getcwd(), "SoundAQnet_event_probability"),
        help="Directory to save event probability outputs."
    )
    parser.add_argument(
        "--paq_output_dir",
        type=str,
        default=os.path.join(os.getcwd(), "SoundAQnet_scene_ISOPl_ISOEv_PAQ8DAQs"),
        help="Directory to save scene / ISO / PAQ outputs."
    )
    args = parser.parse_args()

    node_emb_dim = 64
    hidden_dim, out_dim = 32, 64
    batch_size = 32
    config.batch_size = batch_size
    number_of_nodes = 8
    monitor = "ISOPls"

    using_model = SoundAQnet

    model = using_model(
        max_node_num=number_of_nodes,
        node_emb_dim=node_emb_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim
    )

    event_model_path = args.model
    model_event = torch.load(event_model_path, map_location="cpu")

    if "state_dict" in model_event.keys():
        model.load_state_dict(model_event["state_dict"])
    else:
        model.load_state_dict(model_event)

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

        txtfile = os.path.join(
            result_event_dir, name.replace(".npy", "_event.txt")
        )
        np.savetxt(txtfile, output_dict["output_event"][each_index])
        print("Audio event probability matrix:", output_dict["output_event"][each_index])

        txtfile = os.path.join(
            result_PAQ_dir, name.replace(".npy", "_scene_PAQ.txt")
        )

        with open(txtfile, "w") as f:
            max_id = np.argmax(output_dict["output_scene"][each_index])
            print("Acoustic scene:", config.scene_labels[max_id])
            f.write(config.scene_labels[max_id] + "\n")
            f.write(
                str(output_dict["output_ISOPls"][each_index][0]) + "\t" +
                str(output_dict["output_ISOEvs"][each_index][0]) + "\n"
            )
            print(
                "ISOP and ISOE:",
                str(output_dict["output_ISOPls"][each_index][0]) + "\t" +
                str(output_dict["output_ISOEvs"][each_index][0])
            )

            print(
                "PAQ 8D AQs:",
                str(output_dict["output_pleasant"][each_index][0]) + "\t" +
                str(output_dict["output_eventful"][each_index][0]) + "\t" +
                str(output_dict["output_chaotic"][each_index][0]) + "\t" +
                str(output_dict["output_vibrant"][each_index][0]) + "\t" +
                str(output_dict["output_uneventful"][each_index][0]) + "\t" +
                str(output_dict["output_calm"][each_index][0]) + "\t" +
                str(output_dict["output_annoying"][each_index][0]) + "\t" +
                str(output_dict["output_monotonous"][each_index][0])
            )

            f.write(
                str(output_dict["output_pleasant"][each_index][0]) + "\t" +
                str(output_dict["output_eventful"][each_index][0]) + "\t" +
                str(output_dict["output_chaotic"][each_index][0]) + "\t" +
                str(output_dict["output_vibrant"][each_index][0]) + "\t" +
                str(output_dict["output_uneventful"][each_index][0]) + "\t" +
                str(output_dict["output_calm"][each_index][0]) + "\t" +
                str(output_dict["output_annoying"][each_index][0]) + "\t" +
                str(output_dict["output_monotonous"][each_index][0]) + "\n"
            )

        print("\n")


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)