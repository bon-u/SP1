from flask import Flask
def create_app():
    app = Flask(__name__)

    from .views.main import main
    from .views.indata import indata

    app.register_blueprint(main)
    app.register_blueprint(indata)

    return app
    

if __name__ == "__main__":
    app=create_app()
    app.run()