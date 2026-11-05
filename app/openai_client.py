import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_environment_variables():
	"""
	Load environment variables from a .env file located in the project root directory.
	"""
	env_file = PROJECT_ROOT / ".env"

	load_dotenv(dotenv_path=env_file, override=False)  # Load environment variables from the .env file, restrict overriding existing one
	
def create_openai_client() -> OpenAI:
	"""
	Create an OpenAI client instance.

	Args:
		api_key (str): The API key for authenticating with the OpenAI API.

	Returns:
		OpenAI: An instance of the OpenAI client.
	"""
	load_environment_variables()
	api_key = os.getenv("OPENAI_API_KEY")  # Get the API key from the environment variable

	if not api_key or not api_key.strip():  # Check if the API key is None or empty
		raise RuntimeError("OPENAI_API_KEY is missing. "
            "Configure your API key before creating the client.")

	client = OpenAI(api_key=api_key.strip())
	return client

def get_openai_model() -> str:
	"""
	Retrieve the OpenAI model name from the environment variable.

	Returns:
		str: The OpenAI model name.
	"""
	load_environment_variables()
	model_name = os.getenv("OPENAI_MODEL") 

	if not model_name or not model_name.strip():  # Check if the model name is None or empty
		raise RuntimeError("OPENAI_MODEL is missing. "
			"Configure your model name before using the client.")

	return model_name.strip()