FROM python:3.11.14-slim-bookworm@sha256:65a93d69fa75478d554f4ad27c85c1e69fa184956261b4301ebaf6dbb0a3543d
COPY requirements/provision-linux-amd64-py311.lock.txt /tmp/provision.lock.txt
RUN python -m pip install --no-cache-dir --require-hashes -r /tmp/provision.lock.txt
ENV PYTHONPATH=/usr/src/boptest/service/provision PYTHONUNBUFFERED=1
WORKDIR /usr/src/boptest
CMD ["python", "-m", "boptest_submit", "--shared", "--path", "./testcases/"]
