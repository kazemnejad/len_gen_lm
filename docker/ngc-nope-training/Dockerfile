FROM --platform=linux/arm64 nvcr.io/nvidia/pytorch:23.09-py3

# Install ngrok
RUN apt-get update && \
    apt-get install -y curl && \
    apt-get clean

# Add ngrok's GPG key and debian repository
RUN curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | gpg --dearmor -o /usr/share/keyrings/ngrok-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/ngrok-archive-keyring.gpg] https://ngrok-agent.s3.amazonaws.com buster main" > /etc/apt/sources.list.d/ngrok.list

# Install ngrok
RUN apt-get update && \
    apt-get install -y ngrok && \
    apt-get clean