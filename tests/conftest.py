import pytest
import dotenv
from pathlib import Path

@pytest.fixture(scope = 'session', autouse = True)
def load_env():
    # Load environment before running any test
    dotenv.load_dotenv(
        '.env.prod' if Path('.env.prod').exists() else '.env.dev',
        override = False
    )