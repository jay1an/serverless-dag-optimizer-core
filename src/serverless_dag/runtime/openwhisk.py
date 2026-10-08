"""Small OpenWhisk blocking-invoke client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests
from requests.auth import HTTPBasicAuth
from urllib3.exceptions import InsecureRequestWarning


@dataclass
class OpenWhiskClient:
    apihost: str
    auth: str
    namespace: str = "guest"
    verify_tls: bool = False
    timeout_sec: int = 60

    def __post_init__(self) -> None:
        if ":" not in self.auth:
            raise ValueError("OpenWhisk auth must use the format UUID:SECRET")
        self.uuid, self.secret = self.auth.split(":", 1)
        if not self.apihost:
            raise ValueError("OpenWhisk apihost is required")
        if not self.apihost.startswith(("http://", "https://")):
            self.apihost = f"https://{self.apihost}"
        self.apihost = self.apihost.rstrip("/")
        if not self.verify_tls:
            requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

    def _invoke(
        self, action: str, params: dict[str, Any], result_only: bool
    ) -> dict[str, Any]:
        query = "blocking=true&result=true" if result_only else "blocking=true"
        url = (
            f"{self.apihost}/api/v1/namespaces/{self.namespace}/actions/"
            f"{action}?{query}"
        )
        # Keep OpenWhisk control-plane traffic on the cluster network even when
        # the user's shell exports HTTP(S)_PROXY.  Blocking invokes wait for
        # activation completion, so accidentally routing them through a desktop
        # proxy can turn tiny HTTP overhead into multi-second workflow tails.
        with requests.Session() as session:
            session.trust_env = False
            response = session.post(
                url,
                json=params,
                auth=HTTPBasicAuth(self.uuid, self.secret),
                verify=self.verify_tls,
                timeout=self.timeout_sec,
            )
        if response.status_code >= 400:
            raise RuntimeError(
                "OpenWhisk invoke failed: "
                f"status={response.status_code}, body={response.text[:500]}"
            )
        return response.json()

    def invoke_action(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        return self._invoke(action, params, result_only=True)

    def invoke_activation(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        return self._invoke(action, params, result_only=False)


def activation_annotations(activation: dict[str, Any]) -> dict[str, Any]:
    """Return OpenWhisk activation annotations keyed by annotation name."""

    return {
        annotation.get("key"): annotation.get("value")
        for annotation in activation.get("annotations", [])
        if isinstance(annotation, dict) and annotation.get("key")
    }
