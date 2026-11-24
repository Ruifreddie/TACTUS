"""Regression checks for the benchmark's visibility and averaging rules."""
import unittest
from unittest.mock import Mock

import numpy as np

from baselines.common import metrics
from baselines.common.hull import hull_mask


class MetricTests(unittest.TestCase):
    def test_pck_pools_visible_points_and_excludes_exact_threshold(self):
        export = Mock()
        export.task.side_effect = lambda sid: sid
        gt = np.zeros((16, 3, 2))
        sparse = np.zeros((16, 3), bool)
        sparse[0, 0] = True
        dense = np.zeros_like(sparse)
        dense[0] = True
        export.tool_points.side_effect = [(gt, sparse), (gt, dense)]
        pred = np.zeros((2, 16, 3, 2))
        pred[0, 0, 0, 0] = 12 / 384
        pred[1, 0, :, 0] = np.array([0, 6, 9]) / 384

        how, median = metrics.score_how(export, ["sparse", "dense"], pred)
        self.assertEqual(how["ade"].value(), (12 + 5) / 2)
        self.assertEqual(median, 7.5)
        for threshold, expected in [(4, .25), (8, .5), (12, .75), (16, 1)]:
            self.assertEqual(how[f"pck{threshold}"].value(), expected)

    def test_contact_dice_uses_the_same_frames_for_masks_and_points(self):
        gt = np.tile([[.2, .2], [.8, .2], [.8, .8], [.2, .8]], (16, 1, 1))
        vis = np.zeros((16, 4))
        vis[0, :1] = vis[1, :2] = vis[2, :4] = 1
        mask = hull_mask(gt[0] * 384)
        frame_dice = metrics.contact_frame_dice(gt, vis, pred_masks=[mask] * len(gt))
        export = Mock()
        export.contact_points.return_value = gt, vis
        export.task.return_value = "video"
        points = metrics.score_where(export, ["clip"], ct_xy=[gt])

        for fill in [np.nan, 1.0]:
            cached = np.where(np.isnan(frame_dice), fill, frame_dice)
            masks = metrics.score_where(export, ["clip"], frame_dice=[cached])
            for key in points:
                self.assertEqual(masks[key].n(), 2)
                self.assertEqual(masks[key].value(), points[key].value())

        for invalid in [np.nan, np.inf, -0.1, 1.1]:
            cached = frame_dice.copy()
            cached[1] = invalid
            with self.assertRaisesRegex(ValueError, "clip.*eligible frame"):
                metrics.score_where(export, ["clip"], frame_dice=[cached])


if __name__ == "__main__":
    unittest.main()
