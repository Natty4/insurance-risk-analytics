"""

Configuration settings for the application.

"""
import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
# Define the base directory of the application
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Define the path to the .env file
ENV_FILE = os.path.join(BASE_DIR, '.env')


