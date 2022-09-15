# Python image
# https://pythonspeed.com/articles/base-image-python-docker-images/
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

# Copy training code to image
WORKDIR "/"
COPY trainer /trainer

# Install python packages
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Set up the entry point for container
ENTRYPOINT ["python", "-m", "trainer.train"]
