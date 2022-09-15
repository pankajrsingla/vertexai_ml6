# Python image
# https://pythonspeed.com/articles/base-image-python-docker-images/
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

# Copy prediction server code to image
COPY ./app /app

# Install python packages
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
