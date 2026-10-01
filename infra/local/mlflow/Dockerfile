# Self-hosted MLflow tracking server. The official mlflow package doesn't
# bundle a Postgres driver or an S3 client, both of which the server needs
# here (Postgres backend store, RustFS S3-compatible artifact store), so
# this is a small custom image rather than a stock one.
#
# sqlalchemy is pinned explicitly (not just left to mlflow's own
# requirement range): an unpinned `pip install` picked up a SQLAlchemy
# release newer than what mlflow==2.17.0's internals were built against,
# which no longer exports `FallbackAsyncAdaptedQueuePool` from
# sqlalchemy.pool -- mlflow's sqlalchemy_store.py imports that name at
# module load time, so the server crash-looped on startup with
# "ImportError: cannot import name 'FallbackAsyncAdaptedQueuePool'".
# 2.0.35 is the SQLAlchemy release contemporary with mlflow 2.17.0.
FROM python:3.11-slim
RUN pip install --no-cache-dir mlflow==2.17.0 sqlalchemy==2.0.35 psycopg2-binary==2.9.9 boto3==1.35.36
EXPOSE 5000
