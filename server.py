import json

from outlier_detection.config import AUTH_KEY
from outlier_detection.logger import logger
from outlier_detection.server_utils import DetectionResponseSchema, RequestSchema, ShiftResponseSchema, make_fastapi_app
from outlier_detection.utils import compute_z_score, detect_outlier, np_json_dumps

app = make_fastapi_app(title="Outlier Detection Service", version="0.1", logger=logger, auth_key=AUTH_KEY)


@app.post("/json/detect", response_model=DetectionResponseSchema)
def model_infer(request: RequestSchema):
    request.check_auth_key()
    response_dict = detect_outlier(request)
    return response_dict


@app.post("/json/shift", response_model=ShiftResponseSchema)
def model_infer(request: RequestSchema):
    request.check_auth_key()
    response_dict = compute_z_score(request)
    return json.loads(np_json_dumps(response_dict))
