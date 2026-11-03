FROM python:3.12-slim

# Create non-root user (uid 1000, gid 1000)
RUN groupadd -g 1000 aegis && \
    useradd -u 1000 -g aegis -m -s /bin/bash aegis

# Install pinned dependencies matching requirements.lock
RUN pip install --no-cache-dir \
    numpy==2.5.4 \
    pandas==3.1.0 \
    scipy==1.18.1 \
    pyarrow==26.0.0

# Setup workspace
RUN mkdir -p /workspace/output && \
    chown -R aegis:aegis /workspace

USER 1000:1000
WORKDIR /workspace

CMD ["python3"]

