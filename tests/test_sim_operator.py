"""Operator routes stay separate from tool responses and never mutate physics for Browse."""

import asyncio
import threading
from types import SimpleNamespace

import httpx
import pytest
from drivingbench.shared.contracts import PROTOCOL, Settings
from drivingbench.sim.episode import EpisodePolicy
from drivingbench.sim.operator import attach_operator
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def app():
    application = FastAPI()
    snapshot = {"pose": [0, 0, 0], "map_pose": [6, -1, 153], "sim_time_s": 2, "speed_mps": 0}
    application.state.sim_native = SimpleNamespace(pose_snapshot=lambda: dict(snapshot))
    application.state.sim_scene = {"name": "Test course", "markers": []}
    application.state.sim_declaration = {"time_scale": 1.0, "restrictions": {}}
    application.state.settings = SimpleNamespace(
        snapshot=lambda: (Settings(speed_limit_mps=1.4), 3)
    )
    application.state.sim_episode = SimpleNamespace(
        summary=lambda: {"touched_ids": ["left-01"], "exit_count": 1}, policy=EpisodePolicy()
    )
    application.state.sim_cameras = SimpleNamespace(
        ready={"ready": True},
        errors={},
        last_metrics={"render_s": 0.02},
        browse=lambda pose: b"test-jpeg",
    )

    @application.get("/observe")
    def observe():
        return {"state": "idle", "images": [{"data": "exact", "camera": "narrow"}]}

    return attach_operator(application)


def test_state_is_operator_only_and_shared_limits_are_current(app):
    with TestClient(app) as client:
        state = client.get("/sim/state").json()
        assert state["ui_version"] == 1 and state["protocol"] == PROTOCOL
        assert state["pose"]["map_pose"] == [6, -1, 153]
        assert state["settings"]["speed_limit_mps"] == 1.4
        assert state["episode"]["touched_ids"] == ["left-01"]
        assert state["declaration"] == {"time_scale": 1.0, "restrictions": {}}
        assert state["vehicle"] == {"length_m": 4.63, "width_m": 1.78}
        assert client.get("/observe").json() == {
            "state": "idle",
            "images": [{"data": "exact", "camera": "narrow"}],
        }
        page = client.get("/sim")
        assert page.status_code == 200 and page.headers["cache-control"] == "no-store"
        assert "fetch('/sim/browse'" in page.text
        assert "getJson('/motion'" in page.text and "getJson('/stop'" in page.text
        assert "getJson('/observe'" in page.text


def test_state_reports_configured_declaration_without_changing_it(app):
    declaration = {"time_scale": 0.5, "restrictions": {"label": "configured by runtime"}}
    app.state.sim_declaration = declaration
    with TestClient(app) as client:
        assert client.get("/sim/state").json()["declaration"] == declaration
    assert app.state.sim_declaration is declaration


def test_browse_passes_map_pose_without_moving_car(app):
    poses = []
    app.state.sim_cameras.browse = lambda pose: poses.append(pose) or b"test-jpeg"
    with TestClient(app) as client:
        response = client.post(
            "/sim/browse",
            json={"x": -7, "y": 2, "heading_deg": 90},
            headers={"X-Sim-Operator-Version": "1"},
        )
        assert response.status_code == 200 and response.content == b"test-jpeg"
        assert response.headers["content-type"] == "image/jpeg"
        assert response.headers["x-sim-view"] == "browse"
        assert poses == [(-7, 2, 90)]
        assert client.get("/sim/state").json()["pose"]["pose"] == [0, 0, 0]


@pytest.mark.parametrize(
    "body",
    [
        {"x": 1, "y": 2},
        {"x": 1, "y": 2, "heading_deg": 3, "local": True},
        {"x": "nan", "y": 2, "heading_deg": 3},
    ],
)
def test_invalid_browse_pose_is_rejected(app, body):
    with TestClient(app) as client:
        assert (
            client.post(
                "/sim/browse", json=body, headers={"X-Sim-Operator-Version": "1"}
            ).status_code
            == 422
        )


def test_stale_controls_and_worker_failure_are_visible(app):
    body = {"x": 1, "y": 2, "heading_deg": 3}
    with TestClient(app) as client:
        assert client.post("/sim/browse", json=body).status_code == 409
        assert (
            client.post(
                "/sim/browse", json=body, headers={"X-Sim-Operator-Version": "0"}
            ).status_code
            == 409
        )

        def failed(pose):
            raise TimeoutError("camera queue busy")

        app.state.sim_cameras.browse = failed
        response = client.post("/sim/browse", json=body, headers={"X-Sim-Operator-Version": "1"})
        assert response.status_code == 503 and "camera queue busy" in response.json()["detail"]


@pytest.mark.asyncio
async def test_waiting_browse_does_not_block_state_reads(app):
    entered, release = threading.Event(), threading.Event()

    def delayed(pose):
        entered.set()
        assert release.wait(3)
        return b"test-jpeg"

    app.state.sim_cameras.browse = delayed
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://sim"
    ) as client:
        pending = asyncio.create_task(
            client.post(
                "/sim/browse",
                json={"x": 1, "y": 2, "heading_deg": 3},
                headers={"X-Sim-Operator-Version": "1"},
            )
        )
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            response = await asyncio.wait_for(client.get("/sim/state"), 0.5)
            assert response.status_code == 200 and not pending.done()
        finally:
            release.set()
        assert (await pending).status_code == 200


def test_drive_displays_only_returned_camera_images_with_individual_ages(app):
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the operator DOM behavior test")
    with TestClient(app) as client:
        page = client.get("/sim").text
    function = page.split("function showDrive(){", 1)[1].split("function setMode(next){", 1)[0]
    script = (
        """
const assert=require('node:assert/strict');
function element(){
  const classes=new Set();
  return {children:[],classList:{add:c=>classes.add(c),remove:c=>classes.delete(c),contains:c=>classes.has(c)},
    append(...nodes){this.children.push(...nodes)},replaceChildren(){this.children=[]}};
}
const ids=Object.fromEntries(['driveViews','view','frameAge','viewLabel'].map(id=>[id,element()]));
const $=id=>ids[id],document={createElement:element};
let lastObserve=null;
"""
        + "function showDrive(){"
        + function
        + """
ids.view.src='blob:browse-image';
lastObserve={images:[
  {camera:'narrow',mime_type:'image/jpeg',data:'narrow-exact',age_s:.12},
  {camera:'wide',mime_type:'image/jpeg',data:'wide-exact',age_s:4.56}
]};
showDrive();
assert.equal(ids.driveViews.children.length,2);
for(const [i,name,data,age] of [[0,'Narrow','narrow-exact','0.12'],[1,'Wide','wide-exact','4.56']]){
  const [caption,image]=ids.driveViews.children[i].children;
  assert.equal(caption.children[0].textContent,name);
  assert.equal(caption.children[1].textContent,`Observed image age ${age} s`);
  assert.equal(image.src,`data:image/jpeg;base64,${data}`);
}
assert(ids.view.classList.contains('hidden'));
assert.equal(ids.view.src,'blob:browse-image');
lastObserve={images:[{camera:'wide',mime_type:'image/jpeg',data:'only-wide'}]};
showDrive();
assert.equal(ids.driveViews.children.length,1);
assert.equal(ids.driveViews.children[0].children[0].children[0].textContent,'Wide');
assert.equal(ids.driveViews.children[0].children[0].children[1].textContent,'Image age unavailable');
lastObserve={images:[]};
showDrive();
assert.equal(ids.driveViews.children.length,0);
assert.equal(ids.frameAge.textContent,'Camera unavailable');
assert(ids.view.classList.contains('hidden'));
"""
    )
    subprocess.run([node, "-e", script], check=True, capture_output=True, text=True, timeout=5)
    browse = page.split("async function queueBrowse(){", 1)[1].split("function element(", 1)[0]
    assert "lastObserve=" not in browse
    assert "$('driveViews')" not in browse
    assert "Browse · narrow camera" in browse
