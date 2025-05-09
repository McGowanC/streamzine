# Start with an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /code

# Copy the requirements file into the container at /code
COPY ./requirements.txt /code/requirements.txt

# Install any needed packages specified in requirements.txt
# Using --no-cache-dir and --default-timeout to be robust
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --default-timeout=100 -r /code/requirements.txt

# Copy the rest of the application code (everything in the current directory) into the container at /code
COPY ./app /code/app

# Define environment variable (Cloud Run will set this, good for local testing too)
ENV PORT=8080

# Command to run the application using Uvicorn
# Use shell form to allow $PORT variable expansion
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT
