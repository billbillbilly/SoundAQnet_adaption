import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np

sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])
from framework.processing import *
from framework.data_generator import *
from framework.models_pytorch import *
from framework.pytorch_utils import count_parameters
import framework.config as config


def cal_auc(targets_event, outputs_event):
    aucs = []
    for i in range(targets_event.shape[0]):
        test_y_auc, pred_auc = targets_event[i, :], outputs_event[i, :]
        if np.sum(test_y_auc):
            test_auc = metrics.roc_auc_score(test_y_auc, pred_auc)
            aucs.append(test_auc)
    final_auc_event_branch = sum(aucs) / len(aucs)
    return final_auc_event_branch


def write_clip_outputs(name, scene_i, event_i, isopl_i, isoev_i,
                       plea_i, even_i, chao_i, vibr_i,
                       unev_i, calm_i, anno_i, mono_i,
                       result_event_dir, result_PAQ_dir):
    """Write the per-clip event and scene/PAQ .txt files for a single clip."""
    event_path = os.path.join(result_event_dir, name.replace(".npy", "_event.txt"))
    np.savetxt(event_path, event_i)

    paq_path = os.path.join(result_PAQ_dir, name.replace(".npy", "_scene_PAQ.txt"))
    with open(paq_path, "w") as f:
        max_id = np.argmax(scene_i)
        f.write(config.scene_labels[max_id] + "\n")
        f.write(str(isopl_i[0]) + "\t" + str(isoev_i[0]) + "\n")
        f.write(
            str(plea_i[0]) + "\t" + str(even_i[0]) + "\t" +
            str(chao_i[0]) + "\t" + str(vibr_i[0]) + "\t" +
            str(unev_i[0]) + "\t" + str(calm_i[0]) + "\t" +
            str(anno_i[0]) + "\t" + str(mono_i[0]) + "\n"
        )


def run_inference_streaming(model, generate_func, cuda,
                            result_event_dir, result_PAQ_dir,
                            total_clips=None):
    """Run the model batch by batch and write each clip's outputs immediately.

    Memory stays bounded to one batch; output files appear continuously;
    a re-run skips clips whose PAQ file already exists (resume support).
    A tqdm progress bar tracks clips processed.
    """
    from tqdm import tqdm

    create_folder(result_event_dir)
    create_folder(result_PAQ_dir)

    done = 0
    pbar = tqdm(total=total_clips, unit="clip", desc="Inference", smoothing=0.05)

    for (batch_x, batch_x_loudness, batch_graph, names) in generate_func:
        batch_x = move_data_to_gpu(batch_x, cuda)
        batch_x_loudness = move_data_to_gpu(batch_x_loudness, cuda)

        model.eval()
        with torch.no_grad():
            scene, event, ISOPls, ISOEvs, \
            pleasant, eventful, chaotic, vibrant, \
            uneventful, calm, annoying, monotonous = model(
                batch_x, batch_x_loudness, batch_graph
            )
            event = F.sigmoid(event)

        scene      = scene.data.cpu().numpy()
        event      = event.data.cpu().numpy()
        ISOPls     = ISOPls.data.cpu().numpy()
        ISOEvs     = ISOEvs.data.cpu().numpy()
        pleasant   = pleasant.data.cpu().numpy()
        eventful   = eventful.data.cpu().numpy()
        chaotic    = chaotic.data.cpu().numpy()
        vibrant    = vibrant.data.cpu().numpy()
        uneventful = uneventful.data.cpu().numpy()
        calm       = calm.data.cpu().numpy()
        annoying   = annoying.data.cpu().numpy()
        monotonous = monotonous.data.cpu().numpy()

        for i, name in enumerate(names):
            write_clip_outputs(
                name, scene[i], event[i], ISOPls[i], ISOEvs[i],
                pleasant[i], eventful[i], chaotic[i], vibrant[i],
                uneventful[i], calm[i], annoying[i], monotonous[i],
                result_event_dir, result_PAQ_dir
            )

        done += len(names)
        pbar.update(len(names))

    pbar.close()
    print("Done. Total written this run: {}".format(done), flush=True)


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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Inference batch size. Clips are bucketed by length, so >1 is safe."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip clips whose scene/PAQ output file already exists."
    )
    args = parser.parse_args()

    node_emb_dim = 64
    hidden_dim, out_dim = 32, 64
    config.batch_size = args.batch_size
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

    result_event_dir = args.event_output_dir
    result_PAQ_dir = args.paq_output_dir

    generator = DataGenerator_Mel_loudness_graph(
        dataset_path, node_emb_dim, number_of_nodes
    )

    # When resuming, tell the generator which clips already have output so it
    # can skip loading them entirely.
    skip_dir = result_PAQ_dir if args.resume else None
    generate_func = generator.generate_inference_soundscape_clip_for_LLM(
        dataset_mel, dataset_loudness, skip_done_dir=skip_dir
    )

    # Count clips up front so the progress bar has a total (and an ETA).
    all_mel = [f for f in os.listdir(dataset_mel) if f.lower().endswith(".npy")]
    if skip_dir is not None:
        total_clips = sum(
            1 for f in all_mel
            if not os.path.exists(os.path.join(skip_dir, f.replace(".npy", "_scene_PAQ.txt")))
        )
    else:
        total_clips = len(all_mel)

    run_inference_streaming(
        model=model,
        generate_func=generate_func,
        cuda=config.cuda,
        result_event_dir=result_event_dir,
        result_PAQ_dir=result_PAQ_dir,
        total_clips=total_clips,
    )


if __name__ == "__main__":
    main(sys.argv)
    
# import os
# import sys
# import argparse
# import torch
# import numpy as np

# sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])
# from framework.processing import *
# from framework.data_generator import *
# from framework.models_pytorch import *
# from framework.pytorch_utils import count_parameters


# def cal_auc(targets_event, outputs_event):
#     aucs = []
#     for i in range(targets_event.shape[0]):
#         test_y_auc, pred_auc = targets_event[i, :], outputs_event[i, :]
#         if np.sum(test_y_auc):
#             test_auc = metrics.roc_auc_score(test_y_auc, pred_auc)
#             aucs.append(test_auc)
#     final_auc_event_branch = sum(aucs) / len(aucs)
#     return final_auc_event_branch


# def main(argv):
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--dataset",
#         type=str,
#         default=os.path.join(os.getcwd(), "Dataset"),
#         help="Path to the Dataset directory used by DataGenerator_Mel_loudness_graph."
#     )
#     parser.add_argument(
#         "--dataset_mel",
#         type=str,
#         required=True,
#         help="Path to the mel feature directory."
#     )
#     parser.add_argument(
#         "--dataset_wav_loudness",
#         type=str,
#         required=True,
#         help="Path to the wav loudness feature directory."
#     )
#     parser.add_argument(
#         "--model",
#         type=str,
#         required=True,
#         help="Path to the model .pth file."
#     )
#     parser.add_argument(
#         "--event_output_dir",
#         type=str,
#         default=os.path.join(os.getcwd(), "SoundAQnet_event_probability"),
#         help="Directory to save event probability outputs."
#     )
#     parser.add_argument(
#         "--paq_output_dir",
#         type=str,
#         default=os.path.join(os.getcwd(), "SoundAQnet_scene_ISOPl_ISOEv_PAQ8DAQs"),
#         help="Directory to save scene / ISO / PAQ outputs."
#     )
#     args = parser.parse_args()

#     node_emb_dim = 64
#     hidden_dim, out_dim = 32, 64
#     batch_size = 1
#     config.batch_size = batch_size
#     number_of_nodes = 8
#     monitor = "ISOPls"

#     using_model = SoundAQnet

#     model = using_model(
#         max_node_num=number_of_nodes,
#         node_emb_dim=node_emb_dim,
#         hidden_dim=hidden_dim,
#         out_dim=out_dim
#     )

#     event_model_path = args.model
#     model_event = torch.load(event_model_path, map_location="cpu")

#     if "state_dict" in model_event.keys():
#         model.load_state_dict(model_event["state_dict"])
#     else:
#         model.load_state_dict(model_event)

#     if config.cuda:
#         model.cuda()

#     dataset_path = args.dataset
#     dataset_mel = args.dataset_mel
#     dataset_loudness = args.dataset_wav_loudness

#     generator = DataGenerator_Mel_loudness_graph(
#         dataset_path, node_emb_dim, number_of_nodes
#     )

#     generate_func = generator.generate_inference_soundscape_clip_for_LLM(
#         dataset_mel, dataset_loudness
#     )
#     output_dict = forward_for_LLM(
#         model=model, generate_func=generate_func, cuda=config.cuda
#     )

#     result_event_dir = args.event_output_dir
#     create_folder(result_event_dir)

#     result_PAQ_dir = args.paq_output_dir
#     create_folder(result_PAQ_dir)

#     for each_index, name in enumerate(output_dict["audio_names"]):
#         print("Soundscape audio clip:", name.split(".npy")[0])

#         txtfile = os.path.join(
#             result_event_dir, name.replace(".npy", "_event.txt")
#         )
#         np.savetxt(txtfile, output_dict["output_event"][each_index])
#         print("Audio event probability matrix:", output_dict["output_event"][each_index])

#         txtfile = os.path.join(
#             result_PAQ_dir, name.replace(".npy", "_scene_PAQ.txt")
#         )

#         with open(txtfile, "w") as f:
#             max_id = np.argmax(output_dict["output_scene"][each_index])
#             print("Acoustic scene:", config.scene_labels[max_id])
#             f.write(config.scene_labels[max_id] + "\n")
#             f.write(
#                 str(output_dict["output_ISOPls"][each_index][0]) + "\t" +
#                 str(output_dict["output_ISOEvs"][each_index][0]) + "\n"
#             )
#             print(
#                 "ISOP and ISOE:",
#                 str(output_dict["output_ISOPls"][each_index][0]) + "\t" +
#                 str(output_dict["output_ISOEvs"][each_index][0])
#             )

#             print(
#                 "PAQ 8D AQs:",
#                 str(output_dict["output_pleasant"][each_index][0]) + "\t" +
#                 str(output_dict["output_eventful"][each_index][0]) + "\t" +
#                 str(output_dict["output_chaotic"][each_index][0]) + "\t" +
#                 str(output_dict["output_vibrant"][each_index][0]) + "\t" +
#                 str(output_dict["output_uneventful"][each_index][0]) + "\t" +
#                 str(output_dict["output_calm"][each_index][0]) + "\t" +
#                 str(output_dict["output_annoying"][each_index][0]) + "\t" +
#                 str(output_dict["output_monotonous"][each_index][0])
#             )

#             f.write(
#                 str(output_dict["output_pleasant"][each_index][0]) + "\t" +
#                 str(output_dict["output_eventful"][each_index][0]) + "\t" +
#                 str(output_dict["output_chaotic"][each_index][0]) + "\t" +
#                 str(output_dict["output_vibrant"][each_index][0]) + "\t" +
#                 str(output_dict["output_uneventful"][each_index][0]) + "\t" +
#                 str(output_dict["output_calm"][each_index][0]) + "\t" +
#                 str(output_dict["output_annoying"][each_index][0]) + "\t" +
#                 str(output_dict["output_monotonous"][each_index][0]) + "\n"
#             )

#         print("\n")


# if __name__ == "__main__":
#     main(sys.argv)