package server

import (
	"fmt"
	"io"
	"slices"
	"sync"
	"time"
)

// Upper bounds of the latency histogram buckets, in seconds.
var latencyBuckets = [...]float64{0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 10}

type routeKey struct {
	route string
	code  int
}

type histogram struct {
	buckets [len(latencyBuckets)]uint64
	count   uint64
	sum     float64
}

// metrics is a small Prometheus-compatible registry: request counts by route
// and status, and a latency histogram by route.
type metrics struct {
	mu       sync.Mutex
	requests map[routeKey]uint64
	latency  map[string]*histogram
}

func newMetrics() *metrics {
	return &metrics{requests: make(map[routeKey]uint64), latency: make(map[string]*histogram)}
}

func (m *metrics) observe(route string, code int, d time.Duration) {
	secs := d.Seconds()
	m.mu.Lock()
	m.requests[routeKey{route, code}]++
	h := m.latency[route]
	if h == nil {
		h = new(histogram)
		m.latency[route] = h
	}
	for i, le := range latencyBuckets {
		if secs <= le {
			h.buckets[i]++
			break
		}
	}
	h.count++
	h.sum += secs
	m.mu.Unlock()
}

func (m *metrics) write(w io.Writer) {
	m.mu.Lock()
	defer m.mu.Unlock()

	keys := make([]routeKey, 0, len(m.requests))
	for k := range m.requests {
		keys = append(keys, k)
	}
	slices.SortFunc(keys, func(a, b routeKey) int {
		if a.route != b.route {
			if a.route < b.route {
				return -1
			}
			return 1
		}
		return a.code - b.code
	})
	fmt.Fprintln(w, "# HELP quiver_http_requests_total HTTP requests by route and status.")
	fmt.Fprintln(w, "# TYPE quiver_http_requests_total counter")
	for _, k := range keys {
		fmt.Fprintf(w, "quiver_http_requests_total{route=%q,code=\"%d\"} %d\n", k.route, k.code, m.requests[k])
	}

	routes := make([]string, 0, len(m.latency))
	for r := range m.latency {
		routes = append(routes, r)
	}
	slices.Sort(routes)
	fmt.Fprintln(w, "# HELP quiver_http_request_duration_seconds HTTP request latency by route.")
	fmt.Fprintln(w, "# TYPE quiver_http_request_duration_seconds histogram")
	for _, r := range routes {
		h := m.latency[r]
		var cum uint64
		for i, le := range latencyBuckets {
			cum += h.buckets[i]
			fmt.Fprintf(w, "quiver_http_request_duration_seconds_bucket{route=%q,le=\"%g\"} %d\n", r, le, cum)
		}
		fmt.Fprintf(w, "quiver_http_request_duration_seconds_bucket{route=%q,le=\"+Inf\"} %d\n", r, h.count)
		fmt.Fprintf(w, "quiver_http_request_duration_seconds_sum{route=%q} %g\n", r, h.sum)
		fmt.Fprintf(w, "quiver_http_request_duration_seconds_count{route=%q} %d\n", r, h.count)
	}
}
