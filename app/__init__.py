from flask import Flask
from config import Config

flask_app = Flask(__name__)
flask_app.config.from_object(Config)

from app import routes
