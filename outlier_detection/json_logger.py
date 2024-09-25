import json
import logging


# Logging setup
class JSONFormatter(logging.Formatter):
    def format(self, record):
        json_record = {
            "message": record.msg,
            "severity": record.levelname,
            "file": record.pathname,
            "lineno": record.lineno,
        }
        record.msg = json.dumps(json_record)
        return super().format(record)


def get_logger(service_name, log_level, json_output=True):
    logger = logging.getLogger(f"{service_name}_logger")
    logger.setLevel(log_level)

    if json_output:
        google_json_handler = logging.StreamHandler()
        google_json_handler.setFormatter(JSONFormatter())
        logger.addHandler(google_json_handler)
    else:
        local_text_handler = logging.StreamHandler()
        local_text_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(local_text_handler)
    return logger
