import os
from typing import *

from huggingface_hub import login
from transformers import AutoTokenizer

INVALID_TOKEN_ID = -100


def call_autotokenizer_with_hf_token(
    model_name: str, hf_token: str = None, **kwargs
) -> AutoTokenizer:
    """Initialize and return an AutoTokenizer with Hugging Face authentication.

    This function attempts to load a tokenizer for the specified model. If authentication
    is required, it will try to login using the provided token or HF_TOKEN environment variable.

    Args:
        model_name (str): The name or path of the pre-trained model/tokenizer
        hf_token (str, optional): Hugging Face authentication token. If None, will try to use HF_TOKEN env var

    Returns:
        AutoTokenizer: The initialized tokenizer for the specified model

    Raises:
        RuntimeError: If authentication fails or HF_TOKEN is not set when required
        Exception: For other errors during tokenizer initialization
    """
    # Get token from environment variable if not provided
    if hf_token is None:
        hf_token = os.getenv("HF_TOKEN")

    # Call AutoTokenizer and handle 401 error with login
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    except Exception as e:
        if "401" in str(e):
            print("Attempting to login to Hugging Face...")
            try:
                assert hf_token is not None, "HF_TOKEN is not set"
                login(token=hf_token)
                print("Successfully logged in, retrying tokenizer download...")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
            except Exception as login_error:
                raise RuntimeError(
                    "Failed to login to Hugging Face. Please check your credentials or login manually using:\n"
                    "$ huggingface-cli login"
                ) from login_error
        raise e
    return tokenizer
