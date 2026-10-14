from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

from atlas.agent.layout import EARTHDATA_TOKEN, resolve_credential
from atlas.data.base import SceneKind
from atlas.data.credentials import EARTHDATA_CREDENTIALS, earthdata_download_headers
from atlas.data.stac import StacApiClient

LPCLOUD_SEARCH = "https://cmr.earthdata.nasa.gov/stac/LPCLOUD/search"
NSIDC_SEARCH = "https://cmr.earthdata.nasa.gov/stac/NSIDC_CPRD/search"


class _CmrEarthdataClient(StacApiClient):
    """CMR STAC search whose granule files sit behind Earthdata Login."""

    required_credentials: ClassVar[tuple[str, ...]] = EARTHDATA_CREDENTIALS
    download_credentials: ClassVar[tuple[str, ...]] = EARTHDATA_CREDENTIALS

    def download_headers(self, href: str) -> Mapping[str, str]:
        """The Earthdata bearer token for DAAC-hosted files; nothing elsewhere."""
        return earthdata_download_headers(href, resolve_credential(EARTHDATA_TOKEN))


class CmrLpcloudClient(_CmrEarthdataClient):
    """NASA CMR STAC on LPCLOUD.

    A 401 or 403 while ``EARTHDATA_TOKEN`` is unset is a source error that
    names that variable and says to set it with ``/key earthdata`` or the
    env var. The same status with a token present names the rejection and
    the status. The value is not included. The check runs after the
    response, not before the request.
    """

    search_url: ClassVar[str] = LPCLOUD_SEARCH


class CmrNsidcClient(_CmrEarthdataClient):
    """NASA CMR STAC on NSIDC_CPRD.

    A 401 or 403 while ``EARTHDATA_TOKEN`` is unset is a source error that
    names that variable and says to set it with ``/key earthdata`` or the
    env var. The same status with a token present names the rejection and
    the status. The value is not included. The check runs after the
    response, not before the request.
    """

    search_url: ClassVar[str] = NSIDC_SEARCH


class GediL2AClient(CmrLpcloudClient):
    default_collection: ClassVar[str] = "GEDI02_A_002"
    satellite: ClassVar[str] = "gedi"
    scene_kind: ClassVar[SceneKind] = SceneKind.lidar
    nominal_gsd_m: ClassVar[float | None] = 25.0


class ViirsVnp09gaClient(CmrLpcloudClient):
    """VIIRS/NPP surface reflectance daily (VNP09GA)."""

    default_collection: ClassVar[str] = "VNP09GA.v002"
    satellite: ClassVar[str] = "viirs"
    scene_kind: ClassVar[SceneKind] = SceneKind.optical
    nominal_gsd_m: ClassVar[float | None] = 500.0


class IceSat2Atl03Client(CmrNsidcClient):
    default_collection: ClassVar[str] = "ATL03_007"
    satellite: ClassVar[str] = "icesat2"
    scene_kind: ClassVar[SceneKind] = SceneKind.lidar


class SmapL3PassiveClient(CmrNsidcClient):
    default_collection: ClassVar[str] = "SPL3SMP_E_006"
    satellite: ClassVar[str] = "smap"
    scene_kind: ClassVar[SceneKind] = SceneKind.atmosphere
    nominal_gsd_m: ClassVar[float | None] = 9000.0
