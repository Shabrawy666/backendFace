import os

class Config:
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'postgresql://postgres:NraeRTIAGwBMQoAJXbzJhmqKtSwVxYCQ@centerbeam.proxy.rlwy.net:52150/railway')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
