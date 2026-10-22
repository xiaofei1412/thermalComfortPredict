FROM golang:1.24.2-bookworm@sha256:79390b5e5af9ee6e7b1173ee3eac7fadf6751a545297672916b59bfa0ecf6f71 AS build
COPY minio.tar.gz mc.tar.gz /src/
RUN mkdir /src/minio /src/mc && tar -xzf /src/minio.tar.gz -C /src/minio --strip-components=1 && tar -xzf /src/mc.tar.gz -C /src/mc --strip-components=1
WORKDIR /src/minio
RUN CGO_ENABLED=0 go build -trimpath -o /minio .
WORKDIR /src/mc
RUN CGO_ENABLED=0 go build -trimpath -o /mc .
FROM debian:bookworm-slim@sha256:7ff6ba31527b9e336162fa061746c366c4b2032f4d2cff2078bea43f78efe3f8
COPY --from=build /minio /mc /usr/local/bin/
ENTRYPOINT ["minio"]
