"""Математические проверки разных уровней weighting и overlap policies."""

import unittest

import torch

from starling_ml.ops.losses import (
    generalized_dice_objective,
    overlap_from_probabilities,
    overlap_loss,
    overlap_objective,
    reduce_overlap,
)
from starling_ml.ops.reduction import reduce_pointwise
from starling_ml.ops.weights import inverse_frequency_weights, presence_case_weights


class ReductionTests(unittest.TestCase):
    def test_element_class_and_sample_levels(self):
        loss = torch.tensor(
            [
                [[1.0, 3.0], [2.0, 4.0]],
                [[5.0, 7.0], [6.0, 8.0]],
            ]
        )
        element_weight = torch.tensor(
            [
                [[1.0, 1.0], [1.0, 3.0]],
                [[1.0, 1.0], [1.0, 1.0]],
            ]
        )
        value = reduce_pointwise(
            loss,
            element_weight=element_weight,
            class_weight=torch.tensor([1.0, 2.0]),
            sample_weight=torch.tensor([1.0, 3.0]),
        )
        self.assertAlmostEqual(value.item(), 5.75, places=5)


class OverlapTests(unittest.TestCase):
    def test_empty_target_policies(self):
        logits = torch.zeros(1, 1, 2, 2, requires_grad=True)
        target = torch.zeros_like(logits)

        ignored = overlap_objective(logits, target, empty_target="ignore")
        standard = overlap_objective(logits, target, empty_target="standard")
        soft_fp = overlap_objective(logits, target, empty_target="false_positive")

        self.assertEqual(ignored.item(), 0.0)
        self.assertGreater(standard.item(), soft_fp.item())
        self.assertAlmostEqual(soft_fp.item(), 0.5, places=5)
        (ignored + standard + soft_fp).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_class_weight_is_applied_after_per_class_dice(self):
        prediction = torch.tensor(
            [[
                [[1.0, 1.0], [1.0, 1.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]]
        )
        target = torch.ones_like(prediction)

        favor_good = overlap_from_probabilities(
            prediction,
            target,
            class_weight=torch.tensor([10.0, 1.0]),
            smooth=0.0,
            empty_target="standard",
        )
        favor_bad = overlap_from_probabilities(
            prediction,
            target,
            class_weight=torch.tensor([1.0, 10.0]),
            smooth=0.0,
            empty_target="standard",
        )
        self.assertLess(favor_good.item(), favor_bad.item())

    def test_constant_channel_weight_inside_stats_cancels(self):
        prediction = torch.tensor(
            [[
                [[0.8, 0.2], [0.9, 0.1]],
                [[0.3, 0.7], [0.4, 0.6]],
            ]]
        )
        target = torch.tensor(
            [[
                [[1.0, 0.0], [1.0, 0.0]],
                [[0.0, 1.0], [0.0, 1.0]],
            ]]
        )
        plain = overlap_from_probabilities(
            prediction, target, smooth=0.0, empty_target="standard"
        )
        channel_constant = torch.tensor([2.0, 5.0]).view(1, 2, 1, 1)
        weighted = overlap_from_probabilities(
            prediction,
            target,
            element_weight=channel_constant,
            smooth=0.0,
            empty_target="standard",
        )
        self.assertAlmostEqual(plain.item(), weighted.item(), places=6)

    def test_sample_weight_changes_per_sample_reduction(self):
        prediction = torch.tensor(
            [
                [[[1.0, 1.0], [1.0, 1.0]]],
                [[[0.0, 0.0], [0.0, 0.0]]],
            ]
        )
        target = torch.ones_like(prediction)
        favor_good = overlap_from_probabilities(
            prediction,
            target,
            sample_weight=torch.tensor([10.0, 1.0]),
            smooth=0.0,
            empty_target="standard",
        )
        favor_bad = overlap_from_probabilities(
            prediction,
            target,
            sample_weight=torch.tensor([1.0, 10.0]),
            smooth=0.0,
            empty_target="standard",
        )
        self.assertLess(favor_good.item(), favor_bad.item())

    def test_3d_and_generalized_dice_have_gradients(self):
        logits = torch.randn(2, 3, 4, 5, 6, requires_grad=True)
        target = torch.randint(0, 3, (2, 4, 5, 6))
        value = generalized_dice_objective(
            logits,
            target,
            mode="multiclass",
            class_weight=torch.tensor([1.0, 2.0, 3.0]),
        )
        self.assertTrue(torch.isfinite(value))
        value.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_legacy_empty_target_api(self):
        logits = torch.zeros(2, 1, 4, 4, requires_grad=True)
        target = torch.zeros_like(logits)
        loss, target_sum = overlap_loss(logits, target)
        value = reduce_overlap(loss, target_sum, empty_target="ignore")
        self.assertEqual(value.item(), 0.0)


class WeightTests(unittest.TestCase):
    def test_inverse_frequency_is_normalized_and_clipped(self):
        weights = inverse_frequency_weights(
            [0.5, 0.1], gamma=0.5, min_value=0.5, max_value=2.0
        )
        self.assertAlmostEqual(weights.mean().item(), 1.0, places=5)
        self.assertGreater(weights[1].item(), weights[0].item())

    def test_presence_weights_distinguish_empty_cases(self):
        target = torch.tensor(
            [
                [[[1.0, 0.0]], [[0.0, 0.0]]],
                [[[0.0, 0.0]], [[1.0, 1.0]]],
            ]
        )
        weights = presence_case_weights(
            target,
            present_weight=[2.0, 3.0],
            empty_weight=[0.5, 0.25],
        )
        expected = torch.tensor([[2.0, 0.25], [0.5, 3.0]])
        self.assertTrue(torch.equal(weights, expected))


if __name__ == "__main__":
    unittest.main()
