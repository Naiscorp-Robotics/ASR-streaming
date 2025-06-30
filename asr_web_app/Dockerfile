FROM python:3.9-slim

WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Install ngrok
RUN apt-get update && apt-get install -y curl unzip && \
    curl -Lo /tmp/ngrok.zip https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.zip && \
    unzip /tmp/ngrok.zip -d /usr/local/bin && \
    rm -f /tmp/ngrok.zip

# Create a startup script
RUN echo '#!/bin/bash\n\
if [ -n "$NGROK_AUTHTOKEN" ]; then\n\
  # Start ngrok in the background\n\
  ngrok http 5000 --authtoken $NGROK_AUTHTOKEN &\n\
  # Wait for ngrok to initialize\n\
  sleep 2\n\
  # Print the public URL\n\
  NGROK_URL=$(curl -s http://localhost:4040/api/tunnels | grep -o "https://[^\"]*")\n\
  echo "==> Ngrok is running! Public URL: $NGROK_URL" \n\
fi\n\
# Start the Flask app\n\
exec python -u app.py\n' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Run the startup script
ENTRYPOINT ["/app/entrypoint.sh"] 