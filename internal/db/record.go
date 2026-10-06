package db

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
)

// WAL record layout, little-endian:
//
//	op u8 | count u32 | entries
//	upsert entry: id u64 | payloadLen u32 | payload JSON | dim * float32
//	delete entry: id u64
//
// The vector length is not stored; it is the collection's dimension.
const (
	opUpsert byte = 1
	opDelete byte = 2
)

var errBadRecord = errors.New("db: malformed wal record")

func encodeUpsert(points []Point, dim int) []byte {
	size := 5
	for _, p := range points {
		size += 12 + len(p.Payload) + 4*dim
	}
	buf := make([]byte, 0, size)
	buf = append(buf, opUpsert)
	buf = binary.LittleEndian.AppendUint32(buf, uint32(len(points)))
	for _, p := range points {
		buf = binary.LittleEndian.AppendUint64(buf, p.ID)
		buf = binary.LittleEndian.AppendUint32(buf, uint32(len(p.Payload)))
		buf = append(buf, p.Payload...)
		for _, x := range p.Vector {
			buf = binary.LittleEndian.AppendUint32(buf, math.Float32bits(x))
		}
	}
	return buf
}

func encodeDelete(ids []uint64) []byte {
	buf := make([]byte, 0, 5+8*len(ids))
	buf = append(buf, opDelete)
	buf = binary.LittleEndian.AppendUint32(buf, uint32(len(ids)))
	for _, id := range ids {
		buf = binary.LittleEndian.AppendUint64(buf, id)
	}
	return buf
}

// entry is one decoded point ready to be applied to the index.
type entry struct {
	id      uint64
	vector  []float32
	payload map[string]any
}

// decodeRecord parses a WAL record. Nothing in the result aliases rec, which
// the log reuses after the replay callback returns.
func decodeRecord(rec []byte, dim int) (op byte, entries []entry, err error) {
	if len(rec) < 5 {
		return 0, nil, errBadRecord
	}
	op = rec[0]
	n := int(binary.LittleEndian.Uint32(rec[1:5]))
	rec = rec[5:]
	switch op {
	case opDelete:
		if len(rec) != 8*n {
			return 0, nil, errBadRecord
		}
		entries = make([]entry, n)
		for i := range entries {
			entries[i].id = binary.LittleEndian.Uint64(rec[8*i:])
		}
		return op, entries, nil
	case opUpsert:
		if n > len(rec)/12 {
			return 0, nil, errBadRecord
		}
		entries = make([]entry, n)
		for i := range entries {
			if len(rec) < 12 {
				return 0, nil, errBadRecord
			}
			entries[i].id = binary.LittleEndian.Uint64(rec)
			plen := int(binary.LittleEndian.Uint32(rec[8:]))
			rec = rec[12:]
			if len(rec) < plen+4*dim {
				return 0, nil, errBadRecord
			}
			if plen > 0 {
				if err := json.Unmarshal(rec[:plen], &entries[i].payload); err != nil {
					return 0, nil, errBadRecord
				}
			}
			rec = rec[plen:]
			v := make([]float32, dim)
			for j := range v {
				v[j] = math.Float32frombits(binary.LittleEndian.Uint32(rec[4*j:]))
			}
			entries[i].vector = v
			rec = rec[4*dim:]
		}
		if len(rec) != 0 {
			return 0, nil, errBadRecord
		}
		return op, entries, nil
	}
	return 0, nil, errBadRecord
}
