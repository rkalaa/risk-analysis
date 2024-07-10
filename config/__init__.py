from .settings import Config, DevelopmentConfig, ProductionConfig, TestingConfig

def get_config(config_name):
    config_dict = {
        'development': DevelopmentConfig,
        'production': ProductionConfig,
        'testing': TestingConfig,
        'default': Config
    }
    return config_dict.get(config_name, Config)