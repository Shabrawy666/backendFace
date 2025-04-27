import os

class Config:
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'postgresql://postgres:NCQBcsbxIOwYHjsiItcTyvNuXpqyQvLM@centerbeam.proxy.rlwy.net:25274/railway')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
