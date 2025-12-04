import re
from urllib.parse import urlparse
from functools import lru_cache
from typing import Tuple, Dict
import joblib
import pandas as pd
from xgboost import XGBClassifier


MODEL_PATH = "C:/Users/styu0/OneDrive/Desktop/25 full/privacy/project/models/xgb_phishing_model.joblib"
FEATURE_COLUMNS = [
    "having_IPhaving_IP_Address",
    "URLURL_Length",
    "Shortining_Service",
    "having_At_Symbol",
    "double_slash_redirecting",
    "Prefix_Suffix",
    "having_Sub_Domain",
    "SSLfinal_State",
    "Domain_registeration_length",
    "Favicon",
    "port",
    "HTTPS_token",
    "Request_URL",
    "URL_of_Anchor",
    "Links_in_tags",
    "SFH",
    "Submitting_to_email",
    "Abnormal_URL",
    "Redirect",
    "on_mouseover",
    "RightClick",
    "popUpWidnow",
    "Iframe",
    "age_of_domain",
    "DNSRecord",
    "web_traffic",
    "Page_Rank",
    "Google_Index",
    "Links_pointing_to_page",
    "Statistical_report",
]


# model load
@lru_cache(maxsize=1)
def load_model(model_path: str = MODEL_PATH) -> XGBClassifier:
    model = joblib.load(model_path)
    print(f"[predict] Loaded model from: {model_path}")
    return model


# URL feature extraction
def _ensure_scheme(url: str) -> str:
    """Ensure URL has a scheme, e.g. http://"""
    if not url.startswith(("http://", "https://")):
        return "http://" + url
    return url


def _has_ip(url: str) -> int:
    """Return 1 if URL uses an IP address, else -1."""
    ip_pattern = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    return 1 if re.search(ip_pattern, url) else -1


def _url_length_category(url: str) -> int:
    """
    Map URL length to {-1, 0, 1}
    (following common thresholds used in UCI phishing dataset examples)
    """
    length = len(url)
    if length < 54:
        return -1
    elif length <= 75:
        return 0
    return 1


def _has_at_symbol(url: str) -> int:
    return 1 if "@" in url else -1


def _double_slash_redirecting(url: str) -> int:
    """
    If a '//' appears in the URL path part (after protocol), mark as 1 (suspicious).
    Otherwise -1.
    """
    first = url.find("//")
    second = url.find("//", first + 2)
    if second != -1:
        return 1
    return -1


def _prefix_suffix(domain: str) -> int:
    """Hyphen in domain name is often considered suspicious."""
    return 1 if "-" in domain else -1


def _having_sub_domain(domain: str) -> int:
    """
    Count dots in domain:
        1 dot   -> -1  (normal)
        2 dots  -> 0   (suspicious)
        >=3     -> 1   (phishing-like)
    """
    dot_count = domain.count(".")
    if dot_count <= 1:
        return -1
    elif dot_count == 2:
        return 0
    else:
        return 1


def _shortening_service(url: str) -> int:
    """Check if URL uses common shortening services."""
    services = ["bit.ly", "tinyurl", "goo.gl", "ow.ly", "t.co", "is.gd", "buff.ly"]
    return 1 if any(s in url for s in services) else -1


def _https_token(domain: str) -> int:
    """
    If 'https' appears in the domain part (not protocol), it is suspicious.
    """
    d = domain.lower()
    return 1 if "https" in d and not d.startswith("https://") else -1


def extract_url_features(url: str) -> Dict[str, int]:
    """
    Convert a raw URL string into a dict of features
    that match the trained model's feature names.

    NOTE:
    - Only a subset of features is computed from URL.
    - Other features are filled with 0 as "unknown/suspicious".
    """
    url = _ensure_scheme(url)
    parsed = urlparse(url)
    domain = parsed.netloc
    features = {col: 0 for col in FEATURE_COLUMNS}
    features["having_IPhaving_IP_Address"] = _has_ip(url)
    features["URLURL_Length"] = _url_length_category(url)
    features["Shortining_Service"] = _shortening_service(url)
    features["having_At_Symbol"] = _has_at_symbol(url)
    features["double_slash_redirecting"] = _double_slash_redirecting(url)
    features["Prefix_Suffix"] = _prefix_suffix(domain)
    features["having_Sub_Domain"] = _having_sub_domain(domain)
    features["HTTPS_token"] = _https_token(domain)

    return features



# predict
def predict(url: str) -> Tuple[int, float, Dict[str, int]]:

    model = load_model()
    feature_dict = extract_url_features(url)

    # build dataframe
    X = pd.DataFrame([feature_dict], columns=FEATURE_COLUMNS)

    prob = model.predict_proba(X)[0, 1]
    label = int(prob >= 0.5)

    return label, prob, feature_dict

