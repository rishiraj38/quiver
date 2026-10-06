FROM golang:1.24-alpine AS build
WORKDIR /src
COPY go.mod ./
COPY cmd ./cmd
COPY internal ./internal
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /quiver ./cmd/quiver

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /quiver /quiver
VOLUME /data
EXPOSE 6333
ENTRYPOINT ["/quiver", "-data", "/data"]
