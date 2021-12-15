[This code's documentation lives on the grpc.io site.](https://grpc.io/docs/languages/python/basics)

This is a small demo of a python grpc service that can be built using docker.

To run the server simply execute

```
docker-compose up
```

To run the client simply run
```
python route_guide_client.py
```

and the client should populate the `route_guide_db.json` file and output the data to the terminal.
