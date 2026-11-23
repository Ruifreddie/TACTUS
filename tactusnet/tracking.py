"""Tracking layer: turn the 7 seed points into per-frame tracks by handing them to the annotation server's CoTracker job.

`ServerTracker` writes to an external annotation server (one new, append-only version per job), so nothing in the pyramid calls it unless it is
built explicitly and `annotate(..., track=True)` is asked for.  `DryRunTracker` only builds the request body.
"""
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class JobRef:
    """The server-side job a clip belongs to (what `get_detailed_annotation` returns for it)."""
    job_id: str
    task: str
    version: int
    a0: int            # affordance range start (absolute frame)
    a1: int
    b0: int            # action range start
    b1: int
    tool: str
    action: str


def build_submit_body(job: JobRef, prompt: List[Dict[str, Any]]) -> Dict[str, Any]:
    return dict(job_id=job.job_id, task=job.task, version=job.version, start_frame=job.a0, end_frame=job.a1, action_start_frame=job.b0,
                action_end_frame=job.b1, action=job.action, status="", point_prompt=prompt, tool=job.tool)


class TrackingLayer(Protocol):
    def submit(self, job: JobRef, prompt: List[Dict[str, Any]]) -> Dict[str, Any]: ...


class DryRunTracker:
    def submit(self, job: JobRef, prompt: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"outcome": "dry_run", "body": build_submit_body(job, prompt)}


class ServerTracker:
    """`session` is a requests-like session already logged in to the server (the caller owns TLS pinning and credentials)."""

    def __init__(self, session, base_url: str, poll_s: float = 2.0, timeout_base_s: float = 120.0, timeout_per_frame_s: float = 0.5) -> None:
        self.s, self.base, self.poll_s = session, base_url.rstrip("/"), poll_s
        self.timeout_base_s, self.timeout_per_frame_s = timeout_base_s, timeout_per_frame_s

    def _versions(self, task: str, job_id: str) -> List[Dict[str, Any]]:
        return self.s.post(f"{self.base}/api/list_annotations/", json={"task": task}, timeout=60).json()["annotations"].get(job_id, [])

    def submit(self, job: JobRef, prompt: List[Dict[str, Any]]) -> Dict[str, Any]:
        vs = self._versions(job.task, job.job_id)
        if len(vs) != 1 or vs[0]["status"] != "approved":      # already has a newer version (or was edited): never overwrite
            return {"outcome": "skipped_changed", "versions": [(v["version"], v["status"]) for v in vs]}
        t0 = time.time()
        r = self.s.post(f"{self.base}/api/submit-track-job/", json=build_submit_body(job, prompt), timeout=60)
        if r.status_code != 200:
            return {"outcome": "submit_error", "http": r.status_code, "text": r.text[:200]}
        deadline = t0 + self.timeout_base_s + self.timeout_per_frame_s * (job.a1 - job.a0 + 1)
        while time.time() < deadline:
            time.sleep(self.poll_s)
            valid = [v for v in self._versions(job.task, job.job_id) if v["version"] != -1]
            latest = max(valid, key=lambda v: v["version"])
            if latest["version"] > job.version and latest["status"] != "pending":
                detail = self.s.post(f"{self.base}/api/get_detailed_annotation/", json={"job_id": job.job_id, "version": latest["version"]}, timeout=60).json()
                return {"outcome": "success" if latest["status"] == "success" else "failed", "new_version": latest["version"], "server_s": round(time.time() - t0, 1),
                        "failure_msg": detail.get("failure_msg", ""), "points": detail.get("points", [])}
        return {"outcome": "timeout", "server_s": round(time.time() - t0, 1)}
