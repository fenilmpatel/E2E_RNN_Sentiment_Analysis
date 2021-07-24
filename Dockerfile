FROM python:3.7.9-slim-buster

ENV app /app

WORKDIR $app

COPY . ./

RUN pip install -r requirement.txt
 
CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 wsgi

# CMD ["python","app.py"]

