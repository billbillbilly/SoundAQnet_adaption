"""
soundaqnet.data
===============
Bundled normalization statistics computed from the SoundAQnet training set.

These pickle files (norm_log_mel.pickle, norm_loudness.pickle) are loaded
automatically by DataGenerator_Mel_loudness_graph when the user's local
Dataset directory does not contain pre-computed normalization files.

Files
-----
norm_log_mel.pickle   – mean / std arrays for log-mel spectrograms (shape: 64,)
norm_loudness.pickle  – mean / std arrays for ISO 532-1 loudness features (shape: 1,)
"""
