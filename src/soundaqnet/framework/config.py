import torch

####################################################################################################

# Dynamic device selection — use CUDA when available, otherwise CPU.
# (Original code had cuda = 1 hard-coded, which broke on CPU-only machines.)
cuda = int(torch.cuda.is_available())

training = 1
testing = 1

if cuda:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
else:
    device = torch.device("cpu")

mel_bins = 64
batch_size = 32
lr_init = 5e-4
epochs = 100

endswith = ".pth"

event_labels = [
    "Silence",
    "Human sounds",
    "Wind",
    "Water",
    "Natural sounds",
    "Traffic",
    "Sounds of things",
    "Vehicle",
    "Bird",
    "Outside, rural or natural",
    "Environment and background",
    "Speech",
    "Music",
    "Noise",
    "Animal",
]
scene_labels = [
    "public_square",
    "park",
    "street_traffic",
]
each_emotion_class_num = 1
