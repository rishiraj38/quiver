package wal

import (
	"testing"
	"time"
)

// 520 bytes is about one 128-dimension float32 vector plus an ID.
var benchPayload = make([]byte, 520)

func benchOpen(b *testing.B, policy SyncPolicy) *Log {
	b.Helper()
	l, err := Open(b.TempDir(), Options{Sync: policy})
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { l.Close() })
	return l
}

func report(b *testing.B, l *Log) {
	b.Helper()
	b.StopTimer()
	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "appends/s")
	l.syncMu.Lock()
	fsyncs := l.fsyncs
	l.syncMu.Unlock()
	if l.opts.Sync == SyncAlways && fsyncs > 0 {
		b.ReportMetric(float64(b.N)/float64(fsyncs), "appends/fsync")
	}
}

func benchSerial(b *testing.B, policy SyncPolicy) {
	l := benchOpen(b, policy)
	b.SetBytes(int64(len(benchPayload)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := l.Append(benchPayload); err != nil {
			b.Fatal(err)
		}
	}
	report(b, l)
}

// benchParallel runs GOMAXPROCS*8 appenders. With SyncAlways that is the
// case group commit exists for: many callers each waiting on durability.
func benchParallel(b *testing.B, policy SyncPolicy) {
	l := benchOpen(b, policy)
	b.SetBytes(int64(len(benchPayload)))
	b.SetParallelism(8)
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			if _, err := l.Append(benchPayload); err != nil {
				b.Error(err)
				return
			}
		}
	})
	report(b, l)
}

func BenchmarkAppendSyncAlways(b *testing.B)           { benchSerial(b, SyncAlways) }
func BenchmarkAppendSyncAlwaysParallel(b *testing.B)   { benchParallel(b, SyncAlways) }
func BenchmarkAppendSyncInterval(b *testing.B)         { benchSerial(b, SyncInterval) }
func BenchmarkAppendSyncIntervalParallel(b *testing.B) { benchParallel(b, SyncInterval) }
func BenchmarkAppendSyncNever(b *testing.B)            { benchSerial(b, SyncNever) }
func BenchmarkAppendSyncNeverParallel(b *testing.B)    { benchParallel(b, SyncNever) }

func BenchmarkReplay(b *testing.B) {
	l := benchOpen(b, SyncNever)
	const records = 100_000
	for i := 0; i < records; i++ {
		if _, err := l.Append(benchPayload); err != nil {
			b.Fatal(err)
		}
	}
	b.SetBytes(int64(records * len(benchPayload)))
	b.ResetTimer()
	start := time.Now()
	for i := 0; i < b.N; i++ {
		n := 0
		if err := l.Replay(1, func(uint64, []byte) error { n++; return nil }); err != nil || n != records {
			b.Fatalf("replayed %d, err %v", n, err)
		}
	}
	b.ReportMetric(float64(b.N*records)/time.Since(start).Seconds(), "records/s")
}
