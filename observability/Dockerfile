ARG BASE_IMAGE=grafana/loki:3.6.3
FROM busybox:1.37.0-musl AS probe
FROM ${BASE_IMAGE}
COPY --from=probe /bin/busybox /bin/busybox
