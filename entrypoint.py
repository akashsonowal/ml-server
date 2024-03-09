from src import MLServe, InferenceEngine

if __name__ == "__main__":
    engine = InferenceEngine()
    engine.run_server()