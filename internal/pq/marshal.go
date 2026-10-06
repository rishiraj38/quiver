package pq

import (
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"math"
)

// On-disk layout, all integers little-endian:
//
//	magic    [4]byte  "QVPQ"
//	version  uint32
//	dim      uint32
//	m        uint32
//	ksub     uint32   always 256; stored so a future format can vary it
//	codebook [m*ksub*(dim/m)]float32
//	crc      uint32   CRC-32C of everything before it
const (
	magic         = "QVPQ"
	formatVersion = 1
	headerSize    = 4 + 4*4
	trailerSize   = 4
)

// Castagnoli rather than IEEE: it detects more error patterns at these sizes
// and is hardware accelerated on both arm64 and amd64.
var crcTable = crc32.MakeTable(crc32.Castagnoli)

func (q *Quantizer) MarshalBinary() ([]byte, error) {
	if q.m == 0 {
		return nil, errors.New("pq: cannot marshal an untrained quantizer")
	}
	buf := make([]byte, 0, headerSize+4*len(q.codebooks)+trailerSize)
	buf = append(buf, magic...)
	buf = binary.LittleEndian.AppendUint32(buf, formatVersion)
	buf = binary.LittleEndian.AppendUint32(buf, uint32(q.dim))
	buf = binary.LittleEndian.AppendUint32(buf, uint32(q.m))
	buf = binary.LittleEndian.AppendUint32(buf, ksub)
	for _, f := range q.codebooks {
		buf = binary.LittleEndian.AppendUint32(buf, math.Float32bits(f))
	}
	buf = binary.LittleEndian.AppendUint32(buf, crc32.Checksum(buf, crcTable))
	return buf, nil
}

// UnmarshalBinary replaces q with the quantizer encoded in data. On error q
// is left untouched.
func (q *Quantizer) UnmarshalBinary(data []byte) error {
	if len(data) < headerSize+trailerSize {
		return fmt.Errorf("pq: data too short (%d bytes)", len(data))
	}
	if string(data[:4]) != magic {
		return errors.New("pq: bad magic, not a quantizer")
	}
	if v := binary.LittleEndian.Uint32(data[4:]); v != formatVersion {
		return fmt.Errorf("pq: unsupported format version %d", v)
	}

	body, trailer := data[:len(data)-trailerSize], data[len(data)-trailerSize:]
	if want, got := binary.LittleEndian.Uint32(trailer), crc32.Checksum(body, crcTable); want != got {
		return fmt.Errorf("pq: checksum mismatch (stored %08x, computed %08x)", want, got)
	}

	// The checksum only proves the bytes are the ones that were written.
	// The header is still validated so that a well-formed file from a buggy
	// or hostile writer cannot produce a quantizer that panics later.
	dim := uint64(binary.LittleEndian.Uint32(data[8:]))
	m := uint64(binary.LittleEndian.Uint32(data[12:]))
	k := binary.LittleEndian.Uint32(data[16:])
	if k != ksub {
		return fmt.Errorf("pq: unsupported codebook size %d", k)
	}
	if dim == 0 || m == 0 || dim%m != 0 {
		return fmt.Errorf("pq: invalid shape dim=%d m=%d", dim, m)
	}
	// m*ksub*(dim/m) floats; in uint64 so a huge dim cannot overflow.
	floats := ksub * dim
	if uint64(len(body)-headerSize) != 4*floats {
		return fmt.Errorf("pq: codebook is %d bytes, shape dim=%d m=%d needs %d",
			len(body)-headerSize, dim, m, 4*floats)
	}

	codebooks := make([]float32, floats)
	raw := body[headerSize:]
	for i := range codebooks {
		codebooks[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
	}
	*q = Quantizer{dim: int(dim), m: int(m), dsub: int(dim / m), codebooks: codebooks}
	return nil
}
