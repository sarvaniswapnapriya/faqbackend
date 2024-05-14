FROM python:3.10-slim-buster

WORKDIR /app/bot

# Copy the entire content of the current directory into the container's working directory
COPY . /app/

# Update the system and install build essential for installing some Python packages
RUN apt-get update && apt-get install -y build-essential

# Upgrade pip
RUN python -m pip install --upgrade pip

# Copy the requirements file into the container
COPY requirements.txt /app/bot

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 4002
EXPOSE 4002

# Set environment variable
ENV NAME World

# Command to run the application
CMD ["python3", "app.py"]
