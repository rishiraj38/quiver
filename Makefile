.PHONY: build test race bench-data bench fmt

build:
	go build -o bin/quiver ./cmd/quiver
	go build -o bin/quiver-bench ./cmd/quiver-bench

test:
	go test ./...

race:
	go test -race ./...

fmt:
	gofmt -l -w .

bench-data:
	cd bench && uv run --no-project --with-requirements requirements.txt prepare.py

bench: build
	./bin/quiver-bench -data bench/datasets/sift-128-euclidean
