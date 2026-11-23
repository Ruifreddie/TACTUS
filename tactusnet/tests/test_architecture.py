"""Architecture tests: shapes, decode rules, seed selection, hand-over formats, the pyramid's three modes.

CPU only, random weights, a fake encoder and a mock spatial layer: nothing is trained, no GPU, no network.
    python -I tactusnet/tests/test_architecture.py
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tactusnet import (COMBOS, ClipSpec, DryRunTracker, FakeEncoder, JobRef, MockSpatial, SemanticHead, ServerTracker, TactusNet, TemporalLayer,
                        combo_id_of, semantic_loss, tool_action_ids)
from tactusnet import _vendored_heads as V
from tactusnet.labels import ID_CONVENTIONS, split_combo
from tactusnet.schema import SpatialOut, sample_local_indices
from tactusnet.semantic import structured_combo_logits
from tactusnet.seeds import build_prompt, choose_seeds, range_to_local_frames
from tactusnet.tracking import build_submit_body

torch.manual_seed(0)


class LabelTests(unittest.TestCase):
    def test_combo_roundtrip(self):
        for cid, name in enumerate(COMBOS):
            tool, action = split_combo(cid)
            self.assertEqual(combo_id_of(tool, action), cid)

    def test_illegal_pair_is_rejected(self):
        with self.assertRaises(ValueError):
            combo_id_of("hook", "grasp")

    def test_kpt_combo_numbering(self):          # kpt_combo/loss.py: tools hook,grasper,clipper,scissors / actions dissect,grasp,clip,cut
        self.assertEqual([tool_action_ids(c, "kpt_combo") for c in range(6)], [(0, 0), (1, 1), (2, 2), (3, 3), (3, 0), (1, 0)])

    def test_h613_numbering(self):               # the table in the H613 README
        self.assertEqual([tool_action_ids(c, "h613") for c in range(6)], [(0, 0), (1, 1), (3, 3), (2, 2), (2, 0), (1, 0)])

    def test_the_two_conventions_really_differ(self):
        self.assertNotEqual(ID_CONVENTIONS["kpt_combo"]["tool"], ID_CONVENTIONS["h613"]["tool"])


class SeedTests(unittest.TestCase):
    def test_sampling_matches_unitap_rule(self):
        for n in (16, 17, 101, 344, 1672):
            self.assertTrue(np.array_equal(sample_local_indices(n), np.round(np.linspace(0, n - 1, 16)).astype(int)))

    def test_seeds_use_only_frames_inside_the_range_and_are_chosen_separately(self):
        idx = sample_local_indices(160)
        tool = np.full((16, 3), 0.2); contact = np.full((16, 4), 0.05)
        tool[2] = 0.99                        # very confident but outside the range
        tool[9] = 0.6; tool[10] = 0.5         # inside: 9 is best for the tool
        contact[10] = 0.15; contact[9] = 0.10 # inside: 10 is best for the contact
        s, e = int(idx[8]), int(idx[11])
        seeds = choose_seeds(idx, tool, contact, s, e)
        self.assertEqual((seeds.tool_local, seeds.contact_local, seeds.fallback, seeds.inside_n), (int(idx[9]), int(idx[10]), False, 4))

    def test_fallback_when_no_sample_is_inside(self):
        idx = sample_local_indices(160)       # spacing about 10.7 frames
        a = int(idx[5]) + 2
        seeds = choose_seeds(idx, np.ones((16, 3)), np.ones((16, 4)), a, a + 3)
        self.assertTrue(seeds.fallback)
        self.assertIn(seeds.tool_local, (int(idx[5]), int(idx[6])))
        self.assertEqual(seeds.tool_local, seeds.contact_local)

    def test_prompt_format(self):
        sp = MockSpatial().predict(None, (0, 0, 1280, 720), (1280, 720), "hook", "dissect", sample_local_indices(100))
        seeds = choose_seeds(sp.sampled_local, sp.tool_conf, sp.contact_conf, 20, 70)
        prompt = build_prompt(sp, seeds, a0=5000)
        self.assertEqual([p["type"] for p in prompt], ["tool_anchor", "tool_anchor", "tool_tip"] + ["contact"] * 4)
        self.assertTrue(all(0.0 <= p["x"] <= 1.0 and 0.0 <= p["y"] <= 1.0 for p in prompt))
        self.assertEqual({p["frameIndex"] for p in prompt[:3]}, {5000 + seeds.tool_local})
        self.assertEqual({p["frameIndex"] for p in prompt[3:]}, {5000 + seeds.contact_local})

    def test_prompt_is_clamped_into_the_frame(self):
        sp = MockSpatial().predict(None, (0, 0, 100, 100), (100, 100), "hook", "dissect", sample_local_indices(50))
        sp.tool_xy[:] = [-30.0, 250.0]
        seeds = choose_seeds(sp.sampled_local, sp.tool_conf, sp.contact_conf, 0, 49)
        self.assertTrue(all(p["x"] == 0.0 and p["y"] == 1.0 for p in build_prompt(sp, seeds, 0)[:3]))

    def test_range_to_frames(self):
        self.assertEqual(range_to_local_frames(0.0, 15.0, 344), (0, 343))
        self.assertEqual(range_to_local_frames(7.5, 7.5, 101), (50, 50))
        s, e = range_to_local_frames(9.0, 4.0, 200)   # an inverted range is repaired, never returned inverted
        self.assertLessEqual(s, e)
        self.assertEqual(range_to_local_frames(-3.0, 40.0, 100), (0, 99))


class SemanticTests(unittest.TestCase):
    def setUp(self):
        self.tokens = torch.randn(2, 8, 24, 24, 1024)

    def test_dual_and_joint_shapes(self):
        for mode in ("dual", "joint"):
            out = SemanticHead(mode=mode).eval()(self.tokens)
            self.assertEqual(tuple(out["combo_logits"].shape), (2, 6))
            self.assertTrue(torch.isfinite(out["combo_logits"]).all())
        out = SemanticHead(mode="dual").eval()(self.tokens)
        self.assertEqual((tuple(out["tool_logits"].shape), tuple(out["action_logits"].shape)), ((2, 4), (2, 4)))

    def test_composed_logits_pick_the_legal_pair(self):
        tool = torch.full((3, 4), -5.0); action = torch.full((3, 4), -5.0)
        tool[0, 3], action[0, 3] = 5.0, 5.0     # scissors + cut
        tool[1, 3], action[1, 0] = 5.0, 5.0     # scissors + dissect
        tool[2, 1], action[2, 0] = 5.0, 5.0     # grasper + dissect
        self.assertEqual(structured_combo_logits(tool, action).argmax(1).tolist(), [3, 4, 5])

    def test_tool_prior_starts_as_a_no_op(self):
        head = SemanticHead(mode="dual", tool_prior=True).eval()
        with_prior = head(self.tokens)["combo_logits"]
        head.combo_action_tool_prior = None
        self.assertTrue(torch.allclose(with_prior, head(self.tokens)["combo_logits"], atol=1e-6))

    def test_predict_is_a_distribution(self):
        out = SemanticHead().predict(self.tokens[:1])
        self.assertAlmostEqual(sum(out.probs), 1.0, places=4)
        self.assertEqual(out.combo, COMBOS[out.combo_id])
        self.assertEqual(out.source, "predicted")

    def test_parameter_names_match_kpt_combo(self):
        names = {n.split(".")[0] for n, _ in SemanticHead(mode="dual").named_parameters()}
        self.assertEqual(names, {"combo_tool_spatial_pool", "combo_action_temporal_pool", "combo_tool_head", "combo_action_head", "combo_action_tool_prior"})
        self.assertEqual({n.split(".")[0] for n, _ in SemanticHead(mode="joint").named_parameters()}, {"combo_attn_pool", "combo_head"})

    def test_loss_backprops_into_the_heads_only(self):
        head = SemanticHead(mode="dual").train()
        loss = semantic_loss(head(self.tokens), torch.tensor([0, 4]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None for n, p in head.named_parameters() if "tool_prior.alpha" not in n and "tool_to_query" not in n))


class TemporalTests(unittest.TestCase):
    def test_decode_rule(self):                  # the vendored decoder: argmax step, then left/right distance
        act = torch.full((1, 16), -6.0); act[0, 7] = 4.0
        off = torch.zeros(1, 16, 2); off[0, 7] = torch.tensor([3.0, 4.0])
        d = V.decode_range_from_points(act, off)
        self.assertEqual((float(d["pred_action_start_t"][0]), float(d["pred_action_end_t"][0])), (4.0, 11.0))
        self.assertAlmostEqual(float(d["pred_action_start_norm"][0]), 4 / 15, places=6)
        off[0, 7] = torch.tensor([50.0, 50.0])  # clamped into [0, 15]
        d = V.decode_range_from_points(act, off)
        self.assertEqual((float(d["pred_action_start_t"][0]), float(d["pred_action_end_t"][0])), (0.0, 15.0))

    def test_trident_head_outputs(self):
        layer = TemporalLayer(head_mode="trident", proposal_mode="bmn", quality_mode="dgqp").eval()
        out = layer(torch.randn(2, 8, 24, 24, 1024), torch.tensor([0, 3]), torch.tensor([0, 3]), torch.tensor([0, 3]))
        shapes = {k: tuple(v.shape) for k, v in out.items()}
        self.assertEqual(shapes["range_actionness_logits"], (2, 16))
        self.assertEqual(shapes["range_offsets"], (2, 16, 2))
        self.assertEqual(shapes["range_left_dist_logits"], (2, 16, 16))      # 15 bins + 1
        self.assertEqual(shapes["range_proposal_logits"], (2, 16, 16))
        self.assertEqual(shapes["range_quality_logits"], (2, 16))
        self.assertTrue((out["range_offsets"] >= 0).all())

    def test_scalar_head_outputs(self):
        layer = TemporalLayer(head_mode="scalar", proposal_mode="none", quality_mode="none").eval()
        out = layer(torch.randn(1, 8, 24, 24, 1024), torch.tensor([1]), torch.tensor([1]))
        self.assertEqual(tuple(out["range_offsets"].shape), (1, 16, 2))
        self.assertNotIn("range_quality_logits", out)

    def test_condition_changes_the_output(self):
        layer = TemporalLayer().eval(); tok = torch.randn(1, 8, 24, 24, 1024)
        a = layer(tok, torch.tensor([0]), torch.tensor([0]))["range_actionness_logits"]
        b = layer(tok, torch.tensor([3]), torch.tensor([3]))["range_actionness_logits"]
        self.assertFalse(torch.allclose(a, b))

    def test_predict_returns_a_valid_range(self):
        for n in (40, 344):
            out = TemporalLayer().predict(torch.randn(1, 8, 24, 24, 1024), combo_id=3, n_frames=n)
            self.assertTrue(0 <= out.start_local <= out.end_local <= n - 1)
            self.assertTrue(0.0 <= out.start_t <= out.end_t <= 15.0)


class FakeSession:
    """A stand-in for the server: one approved version, then 'pending' for two polls, then 'success'."""

    def __init__(self, versions_before=1):
        self.posts, self.polls, self.versions_before = [], 0, versions_before

    def post(self, url, json=None, timeout=None):
        self.posts.append(url.rsplit("/api/", 1)[-1])
        sess = self

        class R:
            status_code = 200
            text = "null"

            def json(self):
                if url.endswith("/api/list_annotations/"):
                    if sess.versions_before != 1:
                        return {"annotations": {"j1": [{"version": 0, "status": "approved"}, {"version": 1, "status": "success"}]}}
                    sess.polls += 1
                    new = [] if "submit-track-job/" not in " ".join(sess.posts) else [{"version": 1, "status": "pending" if sess.polls < 3 else "success"}]
                    return {"annotations": {"j1": [{"version": 0, "status": "approved"}] + new}}
                if url.endswith("/api/get_detailed_annotation/"):
                    return {"failure_msg": "", "points": [{"x": 1}] * 7}
                return None
        return R()


class TrackingTests(unittest.TestCase):
    JOB = JobRef(job_id="j1", task="case_001_video", version=0, a0=100, a1=199, b0=120, b1=180, tool="hook", action="dissect")
    PROMPT = [{"frameIndex": 150, "x": 0.5, "y": 0.5, "type": "contact"}] * 7

    def test_submit_body_matches_the_server_job_format(self):
        body = build_submit_body(self.JOB, self.PROMPT)
        self.assertEqual(set(body), {"job_id", "task", "version", "start_frame", "end_frame", "action_start_frame", "action_end_frame", "action", "status", "point_prompt", "tool"})
        self.assertEqual(len(body["point_prompt"]), 7)

    def test_dry_run_sends_nothing(self):
        self.assertEqual(DryRunTracker().submit(self.JOB, self.PROMPT)["outcome"], "dry_run")

    def test_server_tracker_submits_once_and_waits_for_success(self):
        s = FakeSession()
        out = ServerTracker(s, "https://x", poll_s=0.0).submit(self.JOB, self.PROMPT)
        self.assertEqual(out["outcome"], "success")
        self.assertEqual(out["new_version"], 1)
        self.assertEqual(s.posts.count("submit-track-job/"), 1)

    def test_server_tracker_never_overwrites_a_job_that_already_has_a_new_version(self):
        s = FakeSession(versions_before=2)
        out = ServerTracker(s, "https://x", poll_s=0.0).submit(self.JOB, self.PROMPT)
        self.assertEqual(out["outcome"], "skipped_changed")
        self.assertNotIn("submit-track-job/", s.posts)


class PyramidTests(unittest.TestCase):
    def setUp(self):
        self.model = TactusNet(FakeEncoder(), SemanticHead(), TemporalLayer(), MockSpatial(), tracker=DryRunTracker())
        self.frames = np.random.RandomState(0).randint(0, 255, (16, 72, 128, 3), dtype=np.uint8)
        self.clip = ClipSpec(a0=14778, n=344)

    def test_full_auto(self):
        ann = self.model.annotate(self.frames, self.clip)
        self.assertEqual((ann.semantic.source, ann.temporal.source), ("predicted", "predicted"))
        js = json.loads(json.dumps(ann.to_json()))
        self.assertEqual(len(js["points"]), 16 * 7)
        self.assertEqual(len(js["prompt"]), 7)
        self.assertTrue(all(14778 <= p["frameIndex"] <= 14778 + 343 for p in js["prompt"]))
        self.assertEqual(len(js["sampled_frames"]), 16)

    def test_semi_auto_keeps_the_given_class(self):
        ann = self.model.annotate(self.frames, self.clip, tool="scissors", action="cut")
        self.assertEqual((ann.semantic.combo_id, ann.semantic.source, ann.temporal.source), (3, "given", "predicted"))

    def test_silver_mode_uses_the_given_range_for_the_seeds(self):
        ann = self.model.annotate(self.frames, self.clip, tool="hook", action="dissect", action_range=(120, 250))
        self.assertEqual((ann.temporal.source, ann.temporal.start_local, ann.temporal.end_local), ("given", 120, 250))
        self.assertTrue(120 <= ann.seeds.tool_local <= 250 and 120 <= ann.seeds.contact_local <= 250)
        self.assertFalse(ann.seeds.fallback)

    def test_tracking_is_opt_in(self):
        job = JobRef("j1", "t", 0, self.clip.a0, self.clip.a1, 120, 250, "hook", "dissect")
        self.assertIsNone(self.model.annotate(self.frames, self.clip, tool="hook", action="dissect", action_range=(120, 250), job=job).tracks)
        ann = self.model.annotate(self.frames, self.clip, tool="hook", action="dissect", action_range=(120, 250), job=job, track=True)
        self.assertEqual(ann.tracks["outcome"], "dry_run")
        self.assertEqual(len(ann.tracks["body"]["point_prompt"]), 7)

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            self.model.annotate(self.frames[:15], self.clip)
        with self.assertRaises(ValueError):
            self.model.annotate(self.frames, self.clip, tool="hook")
        with self.assertRaises(ValueError):
            self.model.annotate(self.frames, self.clip, tool="hook", action="dissect", action_range=(300, 10))
        with self.assertRaises(ValueError):
            self.model.annotate(self.frames, self.clip, tool="hook", action="grasp")
        with self.assertRaises(ValueError):
            self.model.annotate(self.frames, self.clip, track=True)


class VendoredTests(unittest.TestCase):
    def test_vendored_heads_are_in_sync_with_the_zip(self):
        zp = ROOT / "UniTAP_project.zip"
        if not zp.exists():
            self.skipTest("UniTAP_project.zip not present")
        with tempfile.TemporaryDirectory() as d:
            subprocess.run([sys.executable, "-I", str(ROOT / "tactusnet/tools/extract_vendored.py"), "--zip", str(zp), "--out", d], check=True, capture_output=True)
            self.assertEqual((Path(d) / "_vendored_heads.py").read_text(), (ROOT / "tactusnet/_vendored_heads.py").read_text())

    def test_vendored_wrapper_is_byte_identical_to_the_h613_export(self):
        src = ROOT / "repro_exports/affordance_vjepa21_code_20260911/affordance/vjepa_wrapper.py"
        if not src.exists():
            self.skipTest("H613 export not present")
        self.assertEqual(src.read_bytes(), (ROOT / "tactusnet/_vendored_vjepa_wrapper.py").read_bytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
