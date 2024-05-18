# Using Python 3.10 as the base image
FROM python:3.10.14

# Set the working directory in the Docker container
WORKDIR /app

# Copy the Flask app code into the container
COPY ./app /app/app

# Install required Python packages
COPY requirements.txt /app
RUN pip install -r requirements.txt

# Set the PYTHONPATH environment variable to recognize the 'app' package
ENV PYTHONPATH /app/app

# Expose the port the app runs on
EXPOSE 5959

# Define the command to run the Flask application using Waitress
CMD ["waitress-serve", "--listen=0.0.0.0:5959", "--threads=8", "--connection-limit=1000", "main:app"]
