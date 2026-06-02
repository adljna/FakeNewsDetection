"""Compatibility entry point for running the Flask app locally.

The main MLOps service is implemented in src/api.py. This file is kept so the
old command `python app.py` still works.
"""

import os

from src.api import app


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=False)
