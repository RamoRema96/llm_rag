# First stage: setup environment and pull models
FROM python:3.11-slim 

# Install Poetry
RUN pip install poetry

# Update and install curl
RUN apt-get update && apt-get install -y curl

# Install ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy the scripts into the container
COPY start_ollama.sh /usr/local/bin/start_ollama.sh
COPY wait_for_ollama.sh /usr/local/bin/wait_for_ollama.sh

# Make the scripts executable
RUN chmod +x /usr/local/bin/start_ollama.sh
RUN chmod +x /usr/local/bin/wait_for_ollama.sh

# Start the Ollama server, wait for it to be ready, and pull the model
RUN /usr/local/bin/start_ollama.sh & \
    /usr/local/bin/wait_for_ollama.sh && \
    ollama pull llama3.1:latest

COPY . /app

WORKDIR /app

RUN poetry install
RUN chmod +x run_ollama.sh
CMD [ "sh", "run_ollama.sh" ]

#RUN run_ollama.sh


# CMD bash -c "\
#     echo 'Starting Ollama server...'; \
#     ollama serve & \
#     echo 'Waiting for Ollama server to be active...'; \
#     while [ \$(ollama list | grep 'NAME') == '' ]; do \
#         sleep 1; \
#     done; \
#     exec tail -f /dev/null"


# Install various private model
#RUN ollama pull phi3
#RUN ollama pull llama3.1:latest

# Install any other dependencies needed
# For example, install system packages if required
# RUN apt-get update && apt-get install -y ...

# Set the working directory in the container
# WORKDIR /app

# # Copy the pyproject.toml and poetry.lock files first
# COPY pyproject.toml poetry.lock ./

# # Install the Python dependencies using Poetry
# RUN poetry install --no-root

# # Copy the rest of your application code into the container
# COPY . .

# # Example: Copy the model file into the container (adjust as needed)
# # COPY path/to/llama3.1 /path/in/container

# # Specify the command to run your application


# CMD ["poetry", "run", "python", "server.py"]