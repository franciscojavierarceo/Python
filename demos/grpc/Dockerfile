FROM python:3.8

WORKDIR /code

COPY ./* /code/

RUN pip install grpcio
RUN pip install grpcio-tools
CMD ["ls -alh"]

#CMD ["python", "/code/greeter_server.py"]

