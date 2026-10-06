// Command quiver runs the vector database server.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/rishiraj38/quiver/internal/db"
	"github.com/rishiraj38/quiver/internal/server"
	"github.com/rishiraj38/quiver/internal/wal"
)

func main() {
	var (
		addr     = flag.String("addr", defaultAddr(), "listen address (default follows $PORT when set)")
		dataDir  = flag.String("data", "./data", "data directory")
		syncMode = flag.String("sync", "always", "wal durability: always (fsync before acknowledging a write), interval, never")
		syncEach = flag.Duration("sync-interval", 100*time.Millisecond, "fsync period when -sync=interval")
		snapEach = flag.Duration("snapshot-interval", 10*time.Minute, "how often to snapshot and trim the wal; 0 disables")
		readOnly = flag.Bool("read-only", false, "reject writes; for a public demo instance")
	)
	flag.Parse()
	log := slog.New(slog.NewTextHandler(os.Stderr, nil))

	if err := run(log, *addr, *dataDir, *syncMode, *syncEach, *snapEach, *readOnly); err != nil {
		log.Error("fatal", "err", err)
		os.Exit(1)
	}
}

// defaultAddr honours the PORT variable that container platforms inject.
func defaultAddr() string {
	if p := os.Getenv("PORT"); p != "" {
		return ":" + p
	}
	return ":6333"
}

func run(log *slog.Logger, addr, dataDir, syncMode string, syncEach, snapEach time.Duration, readOnly bool) error {
	walOpts := wal.Options{Interval: syncEach}
	switch syncMode {
	case "always":
		walOpts.Sync = wal.SyncAlways
	case "interval":
		walOpts.Sync = wal.SyncInterval
	case "never":
		walOpts.Sync = wal.SyncNever
	default:
		return fmt.Errorf("unknown -sync mode %q", syncMode)
	}

	start := time.Now()
	d, err := db.Open(dataDir, db.Options{WAL: walOpts})
	if err != nil {
		return err
	}
	for _, st := range d.List() {
		log.Info("collection recovered", "name", st.Name, "points", st.Count,
			"snapshot_lsn", st.SnapshotLSN, "last_lsn", st.LastLSN)
	}
	log.Info("database open", "dir", dataDir, "took", time.Since(start).Round(time.Millisecond))

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	if snapEach > 0 {
		go func() {
			t := time.NewTicker(snapEach)
			defer t.Stop()
			for {
				select {
				case <-ctx.Done():
					return
				case <-t.C:
					if err := d.SnapshotAll(); err != nil {
						log.Error("periodic snapshot", "err", err)
					}
				}
			}
		}()
	}

	srv := &http.Server{
		Addr:              addr,
		Handler:           server.New(d, log, server.Options{ReadOnly: readOnly}),
		ReadHeaderTimeout: 10 * time.Second,
	}
	errc := make(chan error, 1)
	go func() { errc <- srv.ListenAndServe() }()
	log.Info("listening", "addr", addr, "sync", syncMode, "read_only", readOnly)

	select {
	case err := <-errc:
		d.Close()
		return err
	case <-ctx.Done():
	}
	log.Info("shutting down")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Error("http shutdown", "err", err)
	}
	// A final snapshot means the next start loads one file instead of
	// replaying the log.
	if err := d.SnapshotAll(); err != nil {
		log.Error("final snapshot", "err", err)
	}
	return d.Close()
}
