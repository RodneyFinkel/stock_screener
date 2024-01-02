FROM python:3.11.3

WORKDIR /app

COPY . /app/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

ENV NAME World

CMD ["streamlit", "run", "app2.py"]


