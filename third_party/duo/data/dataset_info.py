"""Source adapters and their default modality layouts."""

SOURCES = {
    "duplex": "data.duplex_dataset:DuplexDataset",
    "duplex_features": "data.features_dataset:FeatureDataset",
    "raven_features": "data.features_dataset:FeatureDataset",
}
DUPLEX_MODALITIES = {
    "metadata": "{sample_id}/data.json",
    "full_video": "{sample_id}/full.mp4",
    "user_video": "{sample_id}/user_left.mp4",
    "assistant_video": "{sample_id}/assistant_right.mp4",
}
