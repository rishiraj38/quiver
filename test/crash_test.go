// Package test holds process-level tests that run the real server binary.
package test

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math/rand/v2"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"sync/atomic"
	"syscall"
	"testing"
	"time"
)

const crashDim = 32

type node struct {
	cmd  *exec.Cmd
	base string
}

func buildServer(t *testing.T) string {
	t.Helper()
	bin := filepath.Join(t.TempDir(), "quiver")
	out, err := exec.Command("go", "build", "-o", bin, "../cmd/quiver").CombinedOutput()
	if err != nil {
		t.Fatalf("build: %v\n%s", err, out)
	}
	return bin
}

func start(t *testing.T, bin, dataDir string) *node {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := l.Addr().String()
	l.Close()
	cmd := exec.Command(bin, "-addr", addr, "-data", dataDir, "-sync", "always", "-snapshot-interval", "0")
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	n := &node{cmd: cmd, base: "http://" + addr}
	deadline := time.Now().Add(30 * time.Second)
	for time.Now().Before(deadline) {
		resp, err := http.Get(n.base + "/healthz")
		if err == nil {
			resp.Body.Close()
			return n
		}
		time.Sleep(20 * time.Millisecond)
	}
	cmd.Process.Kill()
	t.Fatal("server did not come up")
	return nil
}

func (n *node) kill() {
	n.cmd.Process.Signal(syscall.SIGKILL)
	n.cmd.Wait()
}

func (n *node) post(path string, body any, out any) (int, error) {
	raw, _ := json.Marshal(body)
	resp, err := http.Post(n.base+path, "application/json", bytes.NewReader(raw))
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(resp.Body)
	if out != nil && resp.StatusCode == 200 {
		if err := json.Unmarshal(data, out); err != nil {
			return resp.StatusCode, err
		}
	}
	return resp.StatusCode, nil
}

type point struct {
	ID      uint64         `json:"id"`
	Vector  []float32      `json:"vector"`
	Payload map[string]any `json:"payload"`
}

// The durability contract: once the server has answered 200 to a write, a
// SIGKILL at any later moment must not lose it. Several clients write while
// the process is killed at a random point, the server is restarted on the
// same directory, and every acknowledged point is read back. A snapshot is
// taken on some rounds so recovery covers snapshot + log tail as well.
func TestKillNineLosesNoAcknowledgedWrite(t *testing.T) {
	if testing.Short() {
		t.Skip("runs the server binary")
	}
	bin := buildServer(t)
	dataDir := t.TempDir()

	n := start(t, bin, dataDir)
	req, _ := http.NewRequest("PUT", n.base+"/collections/crash", bytes.NewReader([]byte(fmt.Sprintf(`{"dim":%d}`, crashDim))))
	resp, err := http.DefaultClient.Do(req)
	if err != nil || resp.StatusCode != 201 {
		t.Fatalf("create collection: %v %v", err, resp)
	}
	resp.Body.Close()

	acked := make(map[uint64]float32) // id -> first vector component
	var nextID atomic.Uint64
	rng := rand.New(rand.NewPCG(1, 1))

	const rounds = 6
	for round := range rounds {
		var (
			mu   sync.Mutex
			wg   sync.WaitGroup
			stop atomic.Bool
		)
		for w := range 4 {
			wg.Add(1)
			go func() {
				defer wg.Done()
				r := rand.New(rand.NewPCG(uint64(round), uint64(w)))
				for !stop.Load() {
					batch := make([]point, 1+r.IntN(8))
					for i := range batch {
						v := make([]float32, crashDim)
						for j := range v {
							v[j] = r.Float32()
						}
						batch[i] = point{ID: nextID.Add(1), Vector: v, Payload: map[string]any{"round": round}}
					}
					code, err := n.post("/collections/crash/points", map[string]any{"points": batch}, nil)
					if err != nil || code != 200 {
						// The request raced the kill; it was never acknowledged.
						return
					}
					mu.Lock()
					for _, p := range batch {
						acked[p.ID] = p.Vector[0]
					}
					mu.Unlock()
				}
			}()
		}
		time.Sleep(time.Duration(150+rng.IntN(400)) * time.Millisecond)
		if round%2 == 1 {
			if code, err := n.post("/collections/crash/snapshot", nil, nil); err != nil || code != 200 {
				t.Fatalf("round %d: snapshot: %d %v", round, code, err)
			}
			time.Sleep(time.Duration(50+rng.IntN(200)) * time.Millisecond)
		}
		n.kill()
		stop.Store(true)
		wg.Wait()

		n = start(t, bin, dataDir)
		for id, v0 := range acked {
			resp, err := http.Get(fmt.Sprintf("%s/collections/crash/points/%d", n.base, id))
			if err != nil {
				t.Fatal(err)
			}
			var got point
			json.NewDecoder(resp.Body).Decode(&got)
			resp.Body.Close()
			if resp.StatusCode != 200 || len(got.Vector) != crashDim || got.Vector[0] != v0 {
				n.kill()
				t.Fatalf("round %d: acknowledged point %d lost or changed after kill -9 (status %d)", round, id, resp.StatusCode)
			}
		}
		t.Logf("round %d: %d acknowledged points all present after kill -9", round, len(acked))
	}
	n.kill()
	if len(acked) == 0 {
		t.Fatal("no writes were acknowledged; the test proved nothing")
	}
}
