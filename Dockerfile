# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 7865 available to the world outside this container
EXPOSE 7865

# Run gradio_app.py when the container launches
CMD ["python", "gradio_app.py"]

