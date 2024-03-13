from ml_server import MLServe, ServeImageClassifier

if __name__ == "__main__":
    engine = ServeImageClassifier()
    engine.run_server()