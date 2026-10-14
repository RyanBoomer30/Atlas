"""Asset downloads from Earthdata archives send the Earthdata token, and only there."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from atlas.cli.download import download_assets
from atlas.data.base import Asset, BBox, Scene, SceneKind
from atlas.data.cmr_stac import GediL2AClient
from atlas.data.credentials import earthdata_download_headers, is_earthdata_host
from atlas.data.modis_land_cover import MODISLandCoverClient

LPDAAC = (
    "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/MCD12Q1.061/"
    "MCD12Q1.A2010001.h10v05.061.2022160071156/MCD12Q1.A2010001.h10v05.061.2022160071156.hdf"
)
S3 = "https://lp-prod-protected.s3.us-west-2.amazonaws.com/MCD12Q1.hdf?X-Amz-Signature=abc"
URS = "https://urs.earthdata.nasa.gov/oauth/authorize?client_id=x&app_type=401"
PAYLOAD = b"\x0e\x03\x13\x01hdf-bytes"


@pytest.fixture(autouse=True)
def _no_saved_earthdata_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("EARTHDATA_TOKEN", raising=False)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr("atlas.agent.layout.Path.home", lambda: home)


def _scene(href: str = LPDAAC, source: str = "modis_land_cover") -> Scene:
    return Scene(
        id="G2484151153-LPCLOUD",
        source=source,
        collection="MCD12Q1",
        kind=SceneKind.landcover,
        datetime=datetime(2010, 1, 1, tzinfo=UTC),
        bbox=BBox(west=-104.4, south=29.8, east=-80.6, north=40.1),
        platform="Terra+Aqua",
        assets={"data": Asset(href=href, roles=["data"])},
    )


class Archive:
    """LP DAAC: a token gets a 303 to S3; no token goes to Earthdata Login (401)."""

    def __init__(self, *, accept: str | None = "secret") -> None:
        self.accept = accept
        self.seen: list[tuple[str, str | None]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        auth = request.headers.get("Authorization")
        self.seen.append((request.url.host, auth))
        url = str(request.url)
        if url.startswith(LPDAAC):
            if auth is None:
                return httpx.Response(302, headers={"Location": URS})
            if auth == f"Bearer {self.accept}":
                return httpx.Response(303, headers={"Location": S3})
            return httpx.Response(401, text="invalid token")
        if url.startswith("https://urs.earthdata.nasa.gov/"):
            return httpx.Response(401, text="Unauthorized")
        return httpx.Response(200, content=PAYLOAD, headers={"content-type": "application/x-hdf"})


async def _download(tmp_path: Path, archive: Archive, client: object, scene: Scene) -> list[object]:
    http = httpx.AsyncClient(transport=httpx.MockTransport(archive.handler))
    try:
        report = await download_assets([scene], tmp_path, http=http, clients={scene.source: client})
    finally:
        await http.aclose()
    return report.outcomes


@pytest.mark.asyncio
async def test_modis_download_sends_token_to_lpdaac_and_not_to_s3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EARTHDATA_TOKEN", "secret")
    archive = Archive()
    [outcome] = await _download(tmp_path, archive, MODISLandCoverClient(), _scene())
    assert outcome.status == "saved", outcome.error
    assert outcome.path is not None and outcome.path.read_bytes() == PAYLOAD
    assert archive.seen[0] == ("data.lpdaac.earthdatacloud.nasa.gov", "Bearer secret")
    # httpx drops Authorization on the cross-origin redirect to the presigned URL.
    assert archive.seen[1] == ("lp-prod-protected.s3.us-west-2.amazonaws.com", None)


@pytest.mark.asyncio
async def test_missing_token_names_the_variable_not_the_login_url(tmp_path: Path) -> None:
    archive = Archive()
    [outcome] = await _download(tmp_path, archive, MODISLandCoverClient(), _scene())
    assert outcome.status == "failed"
    assert outcome.error is not None
    assert "EARTHDATA_TOKEN is not set" in outcome.error
    assert "urs.earthdata.nasa.gov" not in outcome.error
    assert not list(tmp_path.rglob("*.hdf"))


@pytest.mark.asyncio
async def test_rejected_token_is_reported_without_its_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EARTHDATA_TOKEN", "expired-token")
    archive = Archive(accept="secret")
    [outcome] = await _download(tmp_path, archive, MODISLandCoverClient(), _scene())
    assert outcome.status == "failed"
    assert outcome.error is not None
    assert "EARTHDATA_TOKEN" in outcome.error and "401" in outcome.error
    assert "expired-token" not in outcome.error


@pytest.mark.asyncio
async def test_cmr_stac_download_sends_token_to_earthdata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EARTHDATA_TOKEN", "secret")
    archive = Archive()
    scene = _scene(source="gedi_l2a")
    [outcome] = await _download(tmp_path, archive, GediL2AClient(), scene)
    assert outcome.status == "saved", outcome.error
    assert archive.seen[0][1] == "Bearer secret"


@pytest.mark.asyncio
async def test_token_is_not_sent_to_other_hosts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EARTHDATA_TOKEN", "secret")
    archive = Archive()
    scene = _scene(href="https://example.org/mirror/MCD12Q1.hdf")
    [outcome] = await _download(tmp_path, archive, MODISLandCoverClient(), scene)
    assert outcome.status == "saved", outcome.error
    assert archive.seen == [("example.org", None)]


@pytest.mark.parametrize(
    ("href", "expected"),
    [
        (LPDAAC, True),
        ("https://data.gesdisc.earthdata.nasa.gov/data/FLDAS/x.nc", True),
        ("https://n5eil01u.ecs.nsidc.org/ATLAS/x.h5", False),
        ("http://data.lpdaac.earthdatacloud.nasa.gov/x.hdf", False),
        ("https://earthdata.nasa.gov.evil.example/x.hdf", False),
        ("https://notearthdata.nasa.gov/x.hdf", False),
    ],
)
def test_earthdata_host_match(href: str, expected: bool) -> None:
    assert is_earthdata_host(href) is expected
    headers = earthdata_download_headers(href, "secret")
    assert ("Authorization" in headers) is expected


def test_no_token_means_no_header() -> None:
    assert earthdata_download_headers(LPDAAC, None) == {}
    assert earthdata_download_headers(LPDAAC, "  ") == {}
