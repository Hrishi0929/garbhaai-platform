# Self-hosted MLflow tracking server. The official mlflow package doesn't
# bundle a Postgres driver or an S3 client, both of which the server needs
# here (Postgres backend store, RustFS S3-compatible artifact store), so
# this is a small custom image rather than a stock one.
FROM python:3.11-slim
RUN pip install --no-cache-dir mlflow==2.17.0 psycopg2-binary==2.9.9 boto3==1.35.36
EXPOSE 5000
