// Package server exposes a DB over a JSON HTTP API.
package server

import (
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strconv"
	"time"

	"github.com/rishiraj38/quiver/internal/db"
)

// maxBody bounds a request body. A batch of 10k 1536-d vectors as JSON is
// roughly 200 MB, so large loads should be split into smaller batches.
const maxBody = 64 << 20

//go:embed ui/index.html
var indexHTML []byte

// Options configures a Server.
type Options struct {
	// ReadOnly rejects every request that would change data, for instances
	// that are open to the internet.
	ReadOnly bool
}

// Server is an http.Handler for one DB.
type Server struct {
	db      *db.DB
	log     *slog.Logger
	opts    Options
	mux     *http.ServeMux
	metrics *metrics
}

func New(d *db.DB, log *slog.Logger, opts Options) *Server {
	s := &Server{db: d, log: log, opts: opts, mux: http.NewServeMux(), metrics: newMetrics()}
	s.mux.HandleFunc("GET /{$}", s.ui)
	s.mux.HandleFunc("GET /healthz", s.health)
	s.mux.HandleFunc("GET /metrics", s.metricsHandler)
	s.mux.HandleFunc("GET /collections", s.listCollections)
	s.mux.HandleFunc("PUT /collections/{name}", s.mutating(s.createCollection))
	s.mux.HandleFunc("GET /collections/{name}", s.getCollection)
	s.mux.HandleFunc("DELETE /collections/{name}", s.mutating(s.dropCollection))
	s.mux.HandleFunc("POST /collections/{name}/points", s.mutating(s.upsert))
	s.mux.HandleFunc("POST /collections/{name}/points/delete", s.mutating(s.deletePoints))
	s.mux.HandleFunc("GET /collections/{name}/points/{id}", s.getPoint)
	s.mux.HandleFunc("POST /collections/{name}/search", s.search)
	s.mux.HandleFunc("POST /collections/{name}/snapshot", s.mutating(s.snapshot))
	return s
}

func (s *Server) mutating(h http.HandlerFunc) http.HandlerFunc {
	if !s.opts.ReadOnly {
		return h
	}
	return func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusForbidden, errorBody{Error: "this server is read-only"})
	}
}

func (s *Server) ui(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write(indexHTML)
}

type statusWriter struct {
	http.ResponseWriter
	code int
}

func (w *statusWriter) WriteHeader(code int) {
	w.code = code
	w.ResponseWriter.WriteHeader(code)
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	sw := &statusWriter{ResponseWriter: w, code: http.StatusOK}
	r.Body = http.MaxBytesReader(sw, r.Body, maxBody)
	s.mux.ServeHTTP(sw, r)
	// The mux records the matched pattern on the request, which keeps the
	// metric label set bounded however many collections exist.
	route := r.Pattern
	if route == "" {
		route = "unmatched"
	}
	s.metrics.observe(route, sw.code, time.Since(start))
}

type errorBody struct {
	Error string `json:"error"`
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(v)
}

func (s *Server) fail(w http.ResponseWriter, r *http.Request, err error) {
	code := http.StatusInternalServerError
	var tooBig *http.MaxBytesError
	switch {
	case errors.Is(err, db.ErrInvalid):
		code = http.StatusBadRequest
	case errors.Is(err, db.ErrNotFound):
		code = http.StatusNotFound
	case errors.Is(err, db.ErrExists):
		code = http.StatusConflict
	case errors.Is(err, db.ErrClosed):
		code = http.StatusServiceUnavailable
	case errors.As(err, &tooBig):
		code = http.StatusRequestEntityTooLarge
	}
	if code == http.StatusInternalServerError {
		s.log.Error("request failed", "method", r.Method, "path", r.URL.Path, "err", err)
	}
	writeJSON(w, code, errorBody{Error: err.Error()})
}

func decode(r *http.Request, v any) error {
	if err := json.NewDecoder(r.Body).Decode(v); err != nil {
		var tooBig *http.MaxBytesError
		if errors.As(err, &tooBig) {
			return err
		}
		return fmt.Errorf("%w: bad JSON body: %v", db.ErrInvalid, err)
	}
	return nil
}

func (s *Server) health(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (s *Server) metricsHandler(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	s.metrics.write(w)
	fmt.Fprintln(w, "# HELP quiver_collection_points Live points per collection.")
	fmt.Fprintln(w, "# TYPE quiver_collection_points gauge")
	stats := s.db.List()
	for _, st := range stats {
		fmt.Fprintf(w, "quiver_collection_points{collection=%q} %d\n", st.Name, st.Count)
	}
	fmt.Fprintln(w, "# HELP quiver_collection_wal_pending Log records not yet covered by a snapshot.")
	fmt.Fprintln(w, "# TYPE quiver_collection_wal_pending gauge")
	for _, st := range stats {
		fmt.Fprintf(w, "quiver_collection_wal_pending{collection=%q} %d\n", st.Name, st.LastLSN-st.SnapshotLSN)
	}
}

func (s *Server) listCollections(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"collections": s.db.List()})
}

func (s *Server) createCollection(w http.ResponseWriter, r *http.Request) {
	var cfg db.CollectionConfig
	if err := decode(r, &cfg); err != nil {
		s.fail(w, r, err)
		return
	}
	c, err := s.db.Create(r.PathValue("name"), cfg)
	if err != nil {
		s.fail(w, r, err)
		return
	}
	writeJSON(w, http.StatusCreated, c.Stats())
}

func (s *Server) getCollection(w http.ResponseWriter, r *http.Request) {
	c, err := s.db.Get(r.PathValue("name"))
	if err != nil {
		s.fail(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, c.Stats())
}

func (s *Server) dropCollection(w http.ResponseWriter, r *http.Request) {
	if err := s.db.Drop(r.PathValue("name")); err != nil {
		s.fail(w, r, err)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func (s *Server) upsert(w http.ResponseWriter, r *http.Request) {
	c, err := s.db.Get(r.PathValue("name"))
	if err != nil {
		s.fail(w, r, err)
		return
	}
	var body struct {
		Points []db.Point `json:"points"`
	}
	if err := decode(r, &body); err != nil {
		s.fail(w, r, err)
		return
	}
	if err := c.Upsert(body.Points); err != nil {
		s.fail(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]int{"upserted": len(body.Points)})
}

func (s *Server) deletePoints(w http.ResponseWriter, r *http.Request) {
	c, err := s.db.Get(r.PathValue("name"))
	if err != nil {
		s.fail(w, r, err)
		return
	}
	var body struct {
		IDs []uint64 `json:"ids"`
	}
	if err := decode(r, &body); err != nil {
		s.fail(w, r, err)
		return
	}
	n, err := c.Delete(body.IDs)
	if err != nil {
		s.fail(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]int{"deleted": n})
}

func (s *Server) getPoint(w http.ResponseWriter, r *http.Request) {
	c, err := s.db.Get(r.PathValue("name"))
	if err != nil {
		s.fail(w, r, err)
		return
	}
	id, err := strconv.ParseUint(r.PathValue("id"), 10, 64)
	if err != nil {
		s.fail(w, r, fmt.Errorf("%w: point id must be an unsigned integer", db.ErrInvalid))
		return
	}
	p, err := c.Get(id)
	if err != nil {
		s.fail(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, p)
}

func (s *Server) search(w http.ResponseWriter, r *http.Request) {
	c, err := s.db.Get(r.PathValue("name"))
	if err != nil {
		s.fail(w, r, err)
		return
	}
	req := db.SearchRequest{WithPayload: true}
	if err := decode(r, &req); err != nil {
		s.fail(w, r, err)
		return
	}
	start := time.Now()
	hits, err := c.Search(req)
	if err != nil {
		s.fail(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"hits":    hits,
		"took_us": time.Since(start).Microseconds(),
	})
}

func (s *Server) snapshot(w http.ResponseWriter, r *http.Request) {
	c, err := s.db.Get(r.PathValue("name"))
	if err != nil {
		s.fail(w, r, err)
		return
	}
	if err := c.Snapshot(); err != nil {
		s.fail(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, c.Stats())
}
