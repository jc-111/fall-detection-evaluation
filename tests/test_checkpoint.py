"""Checkpoint <-> model definition <-> class list compatibility."""
import dataclasses

import pytest
import torch

from fall_detection import _vendor  # noqa: F401
from fall_detection.classifier import ActionClassifier

pytestmark = pytest.mark.models


def test_checkpoint_matches_model_and_class_list(config):
    state = torch.load(config.classifier.weights, map_location="cpu")
    assert state["fcn.weight"].shape[0] == len(config.classifier.class_names) == 9
    clf = ActionClassifier(config.classifier)  # strict load_state_dict inside
    assert clf.class_names == config.classifier.class_names


def test_checkpoint_graph_matches_code_graph(config):
    from Actionsrecognition.Utils import Graph
    state = torch.load(config.classifier.weights, map_location="cpu")
    code_a = torch.tensor(Graph(strategy="spatial").A, dtype=torch.float32)
    for stream in ("pts_stream", "mot_stream"):
        assert torch.allclose(state[f"{stream}.A"], code_a)


def test_wrong_class_list_is_rejected(config):
    bad = dataclasses.replace(config.classifier, class_names=config.classifier.class_names[:7])
    with pytest.raises(RuntimeError):
        ActionClassifier(bad)
