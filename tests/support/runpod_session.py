"""Local HTTP fixtures for the RunPod SessionDriver adapter."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx

from feedbax.orchestration.runpod_session_provider import RunPodCutoffArm, RunPodResources
from feedbax.orchestration.session import ProviderResource, SessionLaunch


def runpod_resources() -> RunPodResources:
    return RunPodResources(
        image=f"runpod/pytorch@sha256:{'a' * 64}", gpu_id="NVIDIA GeForce RTX 4090",
        datacenter="US-TEST-1", cuda_version="12.8", compute_capability="8.9",
        min_driver_major=570, ephemeral_disk_gb=40, hourly_microusd=750_000,
    )


class FakeRunPodCutoff:
    def __init__(self, terminate_argv: tuple[str, ...]):
        self.command = terminate_argv
        self.launches: list[SessionLaunch] = []

    def arm(self, launch: SessionLaunch) -> RunPodCutoffArm:
        self.launches.append(launch)
        return RunPodCutoffArm(
            create_terms={"fixtureRunningCutoffMs": launch.ends_at_ms}, boot_argv=("/usr/bin/true",),
        )

    def terminate_argv(self, resource: ProviderResource, launch: SessionLaunch) -> tuple[str, ...]:
        assert resource.intent == launch.intent
        return self.command


class FakeRunPod:
    pod_id = "pod-1"
    secret_id = "secret-1"

    def __init__(self, resources: RunPodResources, *, on_delete: Callable[[str], None] | None = None):
        self.resources = resources
        self.on_delete = on_delete
        self.pods: dict[str, dict[str, Any]] = {}
        self.secrets: dict[str, dict[str, Any]] = {}
        self.requests: list[tuple[str, str]] = []
        self.created_pod: dict[str, Any] = {}
        self.quoted_rate = resources.hourly_microusd / 1_000_000

    def handle(self, request: httpx.Request) -> httpx.Response:
        method, path = request.method, request.url.path
        self.requests.append((method, path))
        resources = self.resources
        if request.url.host == "auth.docker.io" and (method, path) == ("GET", "/token"):
            return httpx.Response(200, json={"token": "fixture-docker-token"})
        if request.url.host == "registry-1.docker.io" and method == "HEAD":
            return httpx.Response(200, headers={"Docker-Content-Digest": resources.image.split("@")[1]})
        assert request.url.host == "api.runpod.io", request.url
        if (method, path) == ("GET", f"/v2/catalog/gpus/{resources.gpu_id}"):
            return httpx.Response(200, json={
                "id": resources.gpu_id, "price": {"secure": self.quoted_rate},
                "cudaVersions": [{"version": resources.cuda_version, "available": True}],
                "dataCenters": [{"id": resources.datacenter, "availability": "HIGH"}],
            })
        if path == "/v2/pods":
            if method == "GET":
                return httpx.Response(200, json={
                    "pods": list(self.pods.values()), "pagination": {"hasNextPage": False},
                })
            assert method == "POST" and self.pod_id not in self.pods
            self.created_pod = body = json.loads(request.content)
            self.pods[self.pod_id] = {
                "id": self.pod_id, "name": body["name"], "image": body["image"], "cloud": body["cloud"],
                "status": "RUNNING", "disk": body["disk"], "ports": body["ports"],
                "dataCenterId": body["dataCenterIds"][0], "cudaVersion": body["gpu"]["allowedCudaVersions"][0],
                "gpu": {"id": body["gpu"]["id"], "count": body["gpu"]["count"]},
                "mounts": {}, "cost": self.quoted_rate, "createdAt": "2026-11-16T00:00:00Z",
                "ssh": {"direct": {"username": "root", "host": "127.0.0.1", "port": 2222}},
            }
            return httpx.Response(201, json={"id": self.pod_id})
        if path == "/v2/account/secrets":
            if method == "GET":
                name = request.url.params.get("name")
                rows = [row for row in self.secrets.values() if name is None or row["name"] == name]
                return httpx.Response(200, json={"secrets": rows})
            assert method == "POST" and self.secret_id not in self.secrets
            body = json.loads(request.content)
            secret = {"id": self.secret_id, "name": body["name"], "description": body["description"]}
            self.secrets[self.secret_id] = secret
            return httpx.Response(201, json=secret)
        for prefix, rows in (("/v2/pods/", self.pods), ("/v2/account/secrets/", self.secrets)):
            if not path.startswith(prefix):
                continue
            item_id = path.removeprefix(prefix)
            if item_id not in rows:
                return httpx.Response(404)
            if method == "GET":
                return httpx.Response(200, json=rows[item_id])
            assert method == "DELETE"
            del rows[item_id]
            if rows is self.pods and self.on_delete is not None:
                self.on_delete(item_id)
            return httpx.Response(204)
        raise AssertionError(f"Unexpected RunPod fixture request {method} {path}.")
