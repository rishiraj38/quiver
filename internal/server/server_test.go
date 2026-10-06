package server

import (
	"bytes"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/rishiraj38/quiver/internal/db"
	"github.com/rishiraj38/quiver/internal/wal"
)

func newTestServer(t *testing.T) *httptest.Server {
	t.Helper()
	return newTestServerWith(t, Options{})
}

func newTestServerWith(t *testing.T, opts Options) *httptest.Server {
	t.Helper()
	d, err := db.Open(t.TempDir(), db.Options{WAL: wal.Options{Sync: wal.SyncNever}})
	if err != nil {
		t.Fatal(err)
	}
	ts := httptest.NewServer(New(d, slog.New(slog.NewTextHandler(io.Discard, nil)), opts))
	t.Cleanup(func() {
		ts.Close()
		d.Close()
	})
	return ts
}

// call sends a JSON request and decodes the JSON response into out.
func call(t *testing.T, ts *httptest.Server, method, path, body string, out any) int {
	t.Helper()
	req, err := http.NewRequest(method, ts.URL+path, strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := ts.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if out != nil && len(raw) > 0 {
		if err := json.Unmarshal(raw, out); err != nil {
			t.Fatalf("%s %s: bad response %q: %v", method, path, raw, err)
		}
	}
	return resp.StatusCode
}

func TestAPI(t *testing.T) {
	ts := newTestServer(t)

	if code := call(t, ts, "GET", "/healthz", "", nil); code != 200 {
		t.Fatalf("healthz = %d", code)
	}
	if code := call(t, ts, "PUT", "/collections/docs", `{"dim":3,"metric":"cosine"}`, nil); code != 201 {
		t.Fatalf("create = %d", code)
	}
	if code := call(t, ts, "PUT", "/collections/docs", `{"dim":3}`, nil); code != 409 {
		t.Fatalf("duplicate create = %d, want 409", code)
	}

	points := `{"points":[
		{"id":1,"vector":[1,0,0],"payload":{"lang":"go"}},
		{"id":2,"vector":[0.9,0.1,0],"payload":{"lang":"rust"}},
		{"id":3,"vector":[0,1,0],"payload":{"lang":"go"}},
		{"id":4,"vector":[0,0,1]}]}`
	if code := call(t, ts, "POST", "/collections/docs/points", points, nil); code != 200 {
		t.Fatalf("upsert = %d", code)
	}

	var res struct {
		Hits []db.Hit `json:"hits"`
	}
	if code := call(t, ts, "POST", "/collections/docs/search", `{"vector":[1,0,0],"k":2}`, &res); code != 200 {
		t.Fatalf("search = %d", code)
	}
	if len(res.Hits) != 2 || res.Hits[0].ID != 1 || res.Hits[1].ID != 2 {
		t.Fatalf("search hits = %+v", res.Hits)
	}
	if res.Hits[0].Payload["lang"] != "go" {
		t.Fatalf("payload not returned by default: %+v", res.Hits[0])
	}

	filtered := `{"vector":[1,0,0],"k":2,"filter":{"must":[{"key":"lang","op":"eq","value":"go"}]}}`
	res.Hits = nil
	call(t, ts, "POST", "/collections/docs/search", filtered, &res)
	if len(res.Hits) != 2 || res.Hits[0].ID != 1 || res.Hits[1].ID != 3 {
		t.Fatalf("filtered hits = %+v", res.Hits)
	}

	var p db.Hit
	if code := call(t, ts, "GET", "/collections/docs/points/3", "", &p); code != 200 || p.Payload["lang"] != "go" {
		t.Fatalf("get point = %d %+v", code, p)
	}
	var del map[string]int
	if code := call(t, ts, "POST", "/collections/docs/points/delete", `{"ids":[3]}`, &del); code != 200 || del["deleted"] != 1 {
		t.Fatalf("delete = %d %v", code, del)
	}
	if code := call(t, ts, "GET", "/collections/docs/points/3", "", nil); code != 404 {
		t.Fatalf("get deleted point = %d, want 404", code)
	}

	var st db.Stats
	if code := call(t, ts, "POST", "/collections/docs/snapshot", "", &st); code != 200 || st.SnapshotLSN != st.LastLSN || st.Count != 3 {
		t.Fatalf("snapshot = %d %+v", code, st)
	}

	resp, err := ts.Client().Get(ts.URL + "/metrics")
	if err != nil {
		t.Fatal(err)
	}
	raw, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	for _, want := range []string{
		`quiver_collection_points{collection="docs"} 3`,
		`quiver_http_requests_total{route="POST /collections/{name}/search",code="200"} 2`,
		`quiver_http_request_duration_seconds_count{route="POST /collections/{name}/search"} 2`,
	} {
		if !bytes.Contains(raw, []byte(want)) {
			t.Errorf("metrics missing %q", want)
		}
	}

	if code := call(t, ts, "DELETE", "/collections/docs", "", nil); code != 204 {
		t.Fatalf("drop = %d", code)
	}
	if code := call(t, ts, "GET", "/collections/docs", "", nil); code != 404 {
		t.Fatalf("get dropped collection = %d, want 404", code)
	}
}

func TestAPIErrors(t *testing.T) {
	ts := newTestServer(t)
	call(t, ts, "PUT", "/collections/docs", `{"dim":3}`, nil)
	cases := []struct {
		method, path, body string
		want               int
	}{
		{"PUT", "/collections/bad name", `{"dim":3}`, 400},
		{"PUT", "/collections/x", `{"dim":0}`, 400},
		{"PUT", "/collections/x", `{not json`, 400},
		{"POST", "/collections/nope/search", `{"vector":[1,0,0]}`, 404},
		{"POST", "/collections/docs/search", `{"vector":[1,0]}`, 400},
		{"POST", "/collections/docs/search", `{"vector":[1,0,0],"k":1000000}`, 400},
		{"POST", "/collections/docs/search", `{"vector":[1,0,0],"filter":{"must":[{"key":"a","op":"??"}]}}`, 400},
		{"POST", "/collections/docs/points", `{"points":[{"id":1,"vector":[1]}]}`, 400},
		{"POST", "/collections/docs/points", `{"points":[{"id":-1,"vector":[1,0,0]}]}`, 400},
		{"GET", "/collections/docs/points/abc", "", 400},
		{"GET", "/collections/docs/points/7", "", 404},
		{"GET", "/nothing-here", "", 404},
	}
	for _, tc := range cases {
		var e errorBody
		code := call(t, ts, tc.method, tc.path, tc.body, nil)
		if code != tc.want {
			t.Errorf("%s %s %s: got %d, want %d (%s)", tc.method, tc.path, tc.body, code, tc.want, e.Error)
		}
	}
}

func TestReadOnly(t *testing.T) {
	ts := newTestServerWith(t, Options{ReadOnly: true})
	for _, tc := range []struct{ method, path, body string }{
		{"PUT", "/collections/docs", `{"dim":3}`},
		{"DELETE", "/collections/docs", ""},
		{"POST", "/collections/docs/points", `{"points":[]}`},
		{"POST", "/collections/docs/points/delete", `{"ids":[1]}`},
		{"POST", "/collections/docs/snapshot", ""},
	} {
		if code := call(t, ts, tc.method, tc.path, tc.body, nil); code != 403 {
			t.Errorf("%s %s: got %d, want 403", tc.method, tc.path, code)
		}
	}
	if code := call(t, ts, "GET", "/collections", "", nil); code != 200 {
		t.Errorf("list in read-only mode = %d", code)
	}
	// Reads still reach the handler: 404 for the missing collection, not 403.
	if code := call(t, ts, "POST", "/collections/docs/search", `{"vector":[1,0,0]}`, nil); code != 404 {
		t.Errorf("search in read-only mode = %d, want 404", code)
	}
}

func TestUIServed(t *testing.T) {
	ts := newTestServer(t)
	resp, err := ts.Client().Get(ts.URL + "/")
	if err != nil {
		t.Fatal(err)
	}
	raw, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != 200 || !bytes.Contains(raw, []byte("<title>quiver</title>")) {
		t.Fatalf("GET / = %d, %d bytes", resp.StatusCode, len(raw))
	}
}
