import pickle
from transformers import AutoTokenizer, AutoModel
import time
import numpy as np
import torch
import json

from outlier_detection.logger import logger
from outlier_detection.server_utils import RequestSchema


_model: AutoModel = None
_tokenizer: AutoTokenizer = None
_config = None
config_path = "./config/detection_config.pkl"
model_name = 'sentence-transformers/all-MiniLM-L6-v2'


class NumpyAwareJsonEncoder(json.JSONEncoder):
    """Handles numpy datatypes and such in json encoding"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, set):
            return list(obj)
        else:
            return super().default(obj)


def np_json_dumps(data, **kwargs):
    return json.dumps(data, cls=NumpyAwareJsonEncoder, **kwargs)


def load_model():
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        start_time = time.time()
        logger.info(f"Loading model {model_name}")
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModel.from_pretrained(model_name)
        _model.to(device)
        _model.eval()
        logger.info(f"Loaded model {model_name} in {time.time()-start_time:.1f}s")
    return _model, _tokenizer


def load_config():
    global _config
    if _config is None:
        start_time = time.time()
        logger.info(f"Loading config {config_path}")
        with open(config_path, 'rb') as fp:
            _config = pickle.load(fp)
        logger.info(f"Loaded config {config_path} in {time.time()-start_time:.1f}s")
    return _config


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def embed_sentence(text, tokenizer, model):
    encoded_review = tokenizer([text], padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_review_output = model(**encoded_review)

    # Perform pooling
    pooled_review = mean_pooling(model_review_output, encoded_review['attention_mask']).cpu().numpy()

    # Concatenate premise and hypothesis embeddings, as well as their absolute difference
    feature_embeddings = np.array(pooled_review)
    return feature_embeddings[0]


def calculate_euclidean_distance(p1, p2):
    return np.sqrt(np.sum(np.square(p1 - p2)))


def detect_outlier(req: RequestSchema):
    text = f"{req.title} {req.text}"
    category = req.category
    logger.info(f"Running outlier detection for request: {req}")
    start_time = time.time()
    model, tokenizer = load_model()
    config = load_config()
    try:
        embedding = embed_sentence(text.lower(), tokenizer, model)
        centroid_embedding = config[category]
        radius = config["radius"]
        dist = calculate_euclidean_distance(embedding, centroid_embedding)
        detection = dist > radius
        response = {"outlier": detection, "distance": dist,
                    "debug": {"encoder": config["encoder"], "features": config["features"], "radius": config["radius"], "EM": config["EM"], "MV": config["MV"]}}
    except Exception as e:
        logger.error(f"Failed to run outlier detection using {category} centroid: {e}")
        raise e
    logger.info(f"[{time.time() - start_time:.2f}s] Result = {response}")
    return response


def compute_z_score(req: RequestSchema):
    logger.info(f"Scoring distribution shift for request: {req}")
    start_time = time.time()
    text = f"{req.title} {req.text}"
    model, tokenizer = load_model()
    config = load_config()
    try:
        embedding = embed_sentence(text.lower(), tokenizer, model)
        mean_vector = config["mean_vector"]
        std_vector = config["std_vector"]
        # Calculate Z-scores for the new point
        epsilon = 1e-10
        z_scores = (embedding - mean_vector) / (std_vector + epsilon)
        ood_feature_count = (z_scores > 2).sum() + (z_scores < -2).sum()
        mean_z_score = z_scores.mean()
        logger.info(f"[{time.time() - start_time:.2f}s] Mean score = {mean_z_score}, Out of distribution feature count: {ood_feature_count}")
        return {"mean_z_score": mean_z_score, "ood_feature_count": ood_feature_count, "all_z_scores": z_scores,
                "debug": {"encoder": config["encoder"], "features": config["features"], "mean_vector": config["mean_vector"], "std_vector": config["std_vector"]}}
    except Exception as e:
        logger.error(f"Failed to score distribution shift for request {req}: {e}")
        raise e
