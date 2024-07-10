import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Base configuration class."""
    DEBUG = False
    TESTING = False
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
    
    # Database configuration
    DATABASE_URI = os.environ.get('DATABASE_URI') or 'sqlite:///financial_risk_assessment.db'
    
    # API configuration
    API_TITLE = 'Financial Risk Assessment API'
    API_VERSION = '1.0'
    OPENAPI_VERSION = '3.0.2'
    
    # Redis configuration (for caching)
    REDIS_URL = os.environ.get('REDIS_URL') or 'redis://localhost:6379/0'
    
    # Celery configuration (for background tasks)
    CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL') or 'redis://localhost:6379/1'
    CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND') or 'redis://localhost:6379/2'
    
    # Logging configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL') or 'INFO'
    LOG_FILE = os.environ.get('LOG_FILE') or 'financial_risk_assessment.log'
    
    # Financial data API configuration
    FINANCIAL_DATA_API_KEY = os.environ.get('FINANCIAL_DATA_API_KEY')
    FINANCIAL_DATA_API_URL = os.environ.get('FINANCIAL_DATA_API_URL')
    
    # Risk model parameters
    DEFAULT_CONFIDENCE_LEVEL = float(os.environ.get('DEFAULT_CONFIDENCE_LEVEL') or 0.95)
    DEFAULT_TIME_HORIZON = int(os.environ.get('DEFAULT_TIME_HORIZON') or 252)
    
    # Portfolio optimization parameters
    MAX_PORTFOLIO_SIZE = int(os.environ.get('MAX_PORTFOLIO_SIZE') or 50)
    MIN_ASSET_WEIGHT = float(os.environ.get('MIN_ASSET_WEIGHT') or 0.01)
    MAX_ASSET_WEIGHT = float(os.environ.get('MAX_ASSET_WEIGHT') or 0.5)

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """Production configuration."""
    # Production-specific settings
    LOG_LEVEL = 'ERROR'
    
    # Use a more secure SECRET_KEY in production
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'hard-to-guess-secret-key'
    
    # Use a production-ready database
    DATABASE_URI = os.environ.get('DATABASE_URI') or 'postgresql://user:password@localhost/financial_risk_assessment'

class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    DEBUG = True
    
    # Use an in-memory SQLite database for testing
    DATABASE_URI = 'sqlite:///:memory:'
    
    # Disable CSRF tokens in the Forms (only for testing purposes!)
    WTF_CSRF_ENABLED = False