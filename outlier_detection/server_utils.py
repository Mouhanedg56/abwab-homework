import asyncio

from typing import Dict, List, Literal, Optional

from fastapi import Depends, FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from urllib3.exceptions import ReadTimeoutError


def add_exception_handlers(app: FastAPI, logger):
    # Add a more readable error message
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        detail = exc.errors()[0]
        field = list(detail["loc"])
        if field and field[0] == "body":
            field.pop(0)
        if not field or detail["type"] == "value_error.jsondecode":
            error_message = f"Expected a json formatted body."
        else:
            field_loc = ".".join(str(e) for e in field)
            error_message = f"Error for field '{field_loc}': {detail['msg']}"
        if logger:
            logger.warning(f"Error parsing request: {error_message}. Request: {exc.body}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=jsonable_encoder({"raw_error": detail, "error": error_message}),
        )

    @app.exception_handler(PermissionError)
    async def wrong_auth_key(request: Request, exc: PermissionError):
        return make_exception_message("wrong or missing api key", code=status.HTTP_403_FORBIDDEN)

    @app.exception_handler(ConnectionError)
    @app.exception_handler(ReadTimeoutError)
    @app.exception_handler(asyncio.TimeoutError)
    async def downstream_timeout_exception(request: Request, exc: Exception):
        return make_exception_message(
            f"A downstream service can not be reached. Try again later. {exc.__class__.__name__}: {exc}",
            code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    # Log failures to encode response body and return 422
    @app.exception_handler(ValidationError)
    async def response_validation_exception_handler(request, exc):
        if logger:
            logger.error(f"Failed to encode response: {exc}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=jsonable_encoder({"details": str(exc), "error": "pydantic validation error"}),
        )


def make_exception_message(error, code=400):
    return JSONResponse(
        status_code=code,
        content=jsonable_encoder({"error": error, "success": False}),
    )


_auth_key = None


class AuthRequestSchema(BaseModel):
    """Generic base request with simple auth checking"""

    auth_key: str = "wrong"
    environment: str = "Production"

    def check_auth_key(self):
        if self.auth_key != _auth_key:
            raise PermissionError()


def make_fastapi_app(title, version, logger, auth_key=None, utility_endpoints=True):
    """
    logger: your logger, needed for request id adding
    tracing: set up tracing of the requests
    utility_endpoints: add a get /health endpoint
    """
    global _auth_key
    _auth_key = auth_key

    app = FastAPI(title=title, version=version)
    add_exception_handlers(app, logger)

    if utility_endpoints:

        @app.get("/health")  # at the endpoint /health
        def server_health_check():
            return {"status": "healthy"}

    return app


class RequestSchema(AuthRequestSchema):
    title: str
    text: str
    category: Literal["Gift_Cards", "Digital_Music", "Magazine_Subscriptions", "Subscription_Boxes"]


class DetectionResponseSchema(BaseModel):
    outlier: bool
    distance: float
    debug: Optional[Dict] = None


class ShiftResponseSchema(BaseModel):
    mean_z_score: float
    ood_feature_count: int
    all_z_scores: List[float]
    debug: Optional[Dict] = None

