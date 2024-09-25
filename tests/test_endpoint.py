from outlier_detection.config import AUTH_KEY
from outlier_detection.utils import detect_outlier, compute_z_score
from outlier_detection.server_utils import RequestSchema

DETECT_DEBUG_KEYS = {"encoder", "features", "radius", "EM", "MV", "Gift_Cards", "Digital_Music", "Magazine_Subscriptions", "Subscription_Boxes"}
SHIFT_DEBUG_KEYS = {"encoder", "features", "mean_vector", "std_vector"}
DETECT_RESPONSE_KEYS = {"outlier", "distance"}
SHIFT_RESPONSE_KEYS = {"mean_z_score", "mean_z_score", "all_z_scores"}


def test_outlier_detection():
    req = RequestSchema(auth_key=AUTH_KEY, title="", text="Good", category="Gift_Cards")
    result = detect_outlier(req)

    for k in DETECT_RESPONSE_KEYS:
        assert k in result

    for k in DETECT_DEBUG_KEYS:
        assert k in result["debug"]


def test_compute_z_score():
    req = RequestSchema(auth_key=AUTH_KEY, title="", text="Good", category="Gift_Cards")
    result = detect_outlier(req)

    for k in SHIFT_RESPONSE_KEYS:
        assert k in result

    for k in SHIFT_DEBUG_KEYS:
        assert k in result["debug"]


def test_detect_endpoint(client):
    payload = dict(auth_key=AUTH_KEY, title="", text="Good", category="Gift_Cards")
    result = client.post("/json/detect", json=payload).json()
    for k in DETECT_RESPONSE_KEYS:
        assert k in result

    for k in DETECT_DEBUG_KEYS:
        assert k in result["debug"]


def test_shift_endpoint(client):
    payload = dict(auth_key=AUTH_KEY, title="", text="Good", category="Gift_Cards")
    result = client.post("/json/shift", json=payload).json()
    for k in SHIFT_RESPONSE_KEYS:
        assert k in result

    for k in SHIFT_DEBUG_KEYS:
        assert k in result["debug"]
