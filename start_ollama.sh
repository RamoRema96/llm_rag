#!/bin/sh
echo 'Starting Ollama server...'
ollama serve &

exec tail -f /dev/null
