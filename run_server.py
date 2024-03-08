from ml_server.server import app

if __name__ == "__main__":
    import asyncio
    from uvicorn import Config, Server 
    loop = asyncio.new_event_loop()
    config = Config(app=app, loop=loop, host="0.0.0.0", port=8080, log_level="info")y
    uvicorn_server = Server(config)
    loop.run_until_complete(uvicorn_server.serve())