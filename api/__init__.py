from flask import Flask
from flask_restx import Api
from .routes import api as risk_assessment_api

def create_app():
    app = Flask(__name__)
    api = Api(app, version='1.0', title='Financial Risk Assessment API',
              description='A comprehensive API for financial risk assessment and portfolio optimization')
    
    api.add_namespace(risk_assessment_api)
    
    return app