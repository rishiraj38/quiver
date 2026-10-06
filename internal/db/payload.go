package db

import "sync"

const payloadShards = 64

// payloadStore maps point ids to their decoded payloads. It is sharded
// because a filtered search reads it once per visited graph node, from many
// goroutines at once. Stored maps are never mutated, only replaced, so
// readers may hold on to them without copying.
type payloadStore struct {
	shards [payloadShards]payloadShard
}

type payloadShard struct {
	mu sync.RWMutex
	m  map[uint64]map[string]any
}

func (s *payloadStore) shard(id uint64) *payloadShard {
	// Ids are often sequential; the multiply spreads them across shards.
	return &s.shards[(id*0x9E3779B97F4A7C15)>>58]
}

func (s *payloadStore) get(id uint64) map[string]any {
	sh := s.shard(id)
	sh.mu.RLock()
	p := sh.m[id]
	sh.mu.RUnlock()
	return p
}

// set stores p for id; a nil p removes the entry.
func (s *payloadStore) set(id uint64, p map[string]any) {
	sh := s.shard(id)
	sh.mu.Lock()
	if p == nil {
		delete(sh.m, id)
	} else {
		if sh.m == nil {
			sh.m = make(map[uint64]map[string]any)
		}
		sh.m[id] = p
	}
	sh.mu.Unlock()
}

func (s *payloadStore) len() int {
	n := 0
	for i := range s.shards {
		sh := &s.shards[i]
		sh.mu.RLock()
		n += len(sh.m)
		sh.mu.RUnlock()
	}
	return n
}

// each visits every entry. The caller must make sure no writes are running.
func (s *payloadStore) each(fn func(id uint64, p map[string]any) error) error {
	for i := range s.shards {
		for id, p := range s.shards[i].m {
			if err := fn(id, p); err != nil {
				return err
			}
		}
	}
	return nil
}
