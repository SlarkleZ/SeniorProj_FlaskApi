FROM python:3.7.7
WORKDIR /code
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
COPY requirements.txt requirements.txt
RUN pip install webdriver-manager
RUN pip install msedge-selenium-tools selenium==3.141
RUN pip install -r requirements.txt
EXPOSE 5000
COPY . .
CMD ["flask", "run"]