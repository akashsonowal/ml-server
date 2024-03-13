from ml_server import ServeImageClassifier

if __name__ == "__main__":
    engine = ServeImageClassifier()
    engine.run_server()
