FROM pytorch/pytorch

ENV PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=on

WORKDIR /app

# copy and install dependencies first
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r requirements.txt

# copy code and config
COPY . .

# Expose the port the app runs on
EXPOSE 5000

CMD uvicorn server:app --host 0.0.0.0 --port 5000 --workers 1