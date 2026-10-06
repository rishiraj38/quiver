package hnsw

import (
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"math"

	"github.com/rishiraj38/quiver/internal/vec"
)

// ErrCorrupt is wrapped by every error Load returns for a stream that is
// truncated, fails its checksum or describes an impossible graph.
var ErrCorrupt = errors.New("hnsw: corrupt index")

// On-disk layout, all integers little-endian:
//
//	header (44 bytes)
//	    magic "QVHN", version, dim, M, efConstruction, metric   6 x u32
//	    seed                                                    u64
//	    slot count, entry slot                                  2 x u32
//	    CRC32 of the 40 bytes above                             u32
//	then, for every chunk of up to 4096 slots:
//	    one fixed-size record per slot:
//	        id u64, level u8, deleted u8, 2 bytes padding
//	        vector               dim x f32
//	        layer-0 list         (2M+1) x u32: count, neighbours, zero fill
//	    the upper-layer lists of those slots, in slot order:
//	        level x (M+1) x u32 each
//	trailer
//	    CRC32 of everything before it                           u32
//
// The header carries its own checksum because Load sizes its buffers from
// it: a flipped bit in dim must be caught before it turns into a huge
// allocation, not after reading to the end. Records are fixed-size so a
// chunk is two bulk reads of known length; Load therefore never reads past
// the trailer and needs no buffering layer that would.
const (
	magic         = "QVHN"
	formatVersion = 1
	headerSize    = 44
	nodeFixedSize = 12
)

var le = binary.LittleEndian

func (ix *Index) recordSize() int { return nodeFixedSize + 4*ix.dim + 4*ix.stride0 }

// crcWriter counts and checksums what passes through it. The first error
// sticks so callers can check once at the end.
type crcWriter struct {
	w   io.Writer
	sum uint32
	n   int64
	err error
}

func (cw *crcWriter) write(p []byte) {
	if cw.err != nil {
		return
	}
	n, err := cw.w.Write(p)
	cw.n += int64(n)
	cw.sum = crc32.Update(cw.sum, crc32.IEEETable, p[:n])
	cw.err = err
}

// WriteTo serialises the whole index (config, vectors, graph, tombstones) in
// a versioned binary format ending in a CRC32 of everything before it.
func (ix *Index) WriteTo(w io.Writer) (int64, error) {
	// saveMu waits for in-flight inserts to finish linking and holds new
	// ones off; mu does the same for Delete. Searches keep running, and
	// with every writer excluded the lists can be read without node locks.
	ix.saveMu.Lock()
	defer ix.saveMu.Unlock()
	ix.mu.RLock()
	defer ix.mu.RUnlock()

	cw := &crcWriter{w: w}

	var hdr [headerSize]byte
	copy(hdr[0:4], magic)
	le.PutUint32(hdr[4:], formatVersion)
	le.PutUint32(hdr[8:], uint32(ix.dim))
	le.PutUint32(hdr[12:], uint32(ix.cfg.M))
	le.PutUint32(hdr[16:], uint32(ix.cfg.EfConstruction))
	le.PutUint32(hdr[20:], uint32(ix.cfg.Metric))
	le.PutUint64(hdr[24:], uint64(ix.cfg.Seed))
	le.PutUint32(hdr[32:], ix.count)
	entrySlot, _ := unpackEntry(ix.entry.Load())
	if ix.count == 0 {
		entrySlot = 0
	}
	le.PutUint32(hdr[36:], entrySlot)
	le.PutUint32(hdr[40:], crc32.ChecksumIEEE(hdr[:40]))
	cw.write(hdr[:])

	tab := *ix.table.Load()
	rec := ix.recordSize()
	var buf []byte
	for base := 0; base < int(ix.count); base += chunkSize {
		c := tab[base>>chunkShift]
		n := min(chunkSize, int(ix.count)-base)

		buf = resize(buf, n*rec)
		upperWords := 0
		for i := 0; i < n; i++ {
			nd := &c.nodes[i]
			b := buf[i*rec : (i+1)*rec]
			le.PutUint64(b, nd.id)
			b[8] = nd.level
			b[9] = 0
			if nd.deleted.Load() {
				b[9] = 1
			}
			b[10], b[11] = 0, 0
			b = b[nodeFixedSize:]
			for j, f := range c.vecs[i*ix.dim : (i+1)*ix.dim] {
				le.PutUint32(b[4*j:], math.Float32bits(f))
			}
			b = b[4*ix.dim:]
			for j, x := range c.links[i*ix.stride0 : (i+1)*ix.stride0] {
				le.PutUint32(b[4*j:], x)
			}
			upperWords += len(nd.upper)
		}
		cw.write(buf)

		buf = resize(buf, 4*upperWords)
		off := 0
		for i := 0; i < n; i++ {
			for _, x := range c.nodes[i].upper {
				le.PutUint32(buf[off:], x)
				off += 4
			}
		}
		cw.write(buf)
	}
	if cw.err != nil {
		return cw.n, cw.err
	}

	var trailer [4]byte
	le.PutUint32(trailer[:], cw.sum)
	n, err := w.Write(trailer[:])
	return cw.n + int64(n), err
}

func resize(buf []byte, n int) []byte {
	if cap(buf) < n {
		return make([]byte, n)
	}
	return buf[:n]
}

// crcReader checksums what it reads and turns a short read into ErrCorrupt.
type crcReader struct {
	r   io.Reader
	sum uint32
}

func (cr *crcReader) read(p []byte) error {
	if _, err := io.ReadFull(cr.r, p); err != nil {
		return readErr(err)
	}
	cr.sum = crc32.Update(cr.sum, crc32.IEEETable, p)
	return nil
}

func readErr(err error) error {
	if err == io.EOF || err == io.ErrUnexpectedEOF {
		return fmt.Errorf("%w: truncated stream", ErrCorrupt)
	}
	return err
}

func corrupt(format string, args ...any) error {
	return fmt.Errorf("%w: %s", ErrCorrupt, fmt.Sprintf(format, args...))
}

// Load reads an index written by WriteTo and verifies its checksum. It reads
// exactly the bytes WriteTo produced, so an index can be embedded in a larger
// stream. Any error caused by the content of the stream wraps ErrCorrupt.
func Load(r io.Reader) (*Index, error) {
	cr := &crcReader{r: r}

	var hdr [headerSize]byte
	if err := cr.read(hdr[:]); err != nil {
		return nil, err
	}
	if string(hdr[0:4]) != magic {
		return nil, corrupt("bad magic %q", hdr[0:4])
	}
	if v := le.Uint32(hdr[4:]); v != formatVersion {
		return nil, corrupt("unsupported format version %d", v)
	}
	if le.Uint32(hdr[40:]) != crc32.ChecksumIEEE(hdr[:40]) {
		return nil, corrupt("header checksum mismatch")
	}
	dim := le.Uint32(hdr[8:])
	m := le.Uint32(hdr[12:])
	efc := le.Uint32(hdr[16:])
	metric := le.Uint32(hdr[20:])
	seed := int64(le.Uint64(hdr[24:]))
	count := le.Uint32(hdr[32:])
	entrySlot := le.Uint32(hdr[36:])

	// The checksum rules out accidents, not a writer with different limits,
	// so the fields New would have clamped or rejected are still checked.
	switch {
	case dim < 1 || dim > maxDim:
		return nil, corrupt("dimension %d out of range", dim)
	case m < 2 || m > maxM:
		return nil, corrupt("M %d out of range", m)
	case efc < 1 || efc > math.MaxInt32:
		return nil, corrupt("efConstruction %d out of range", efc)
	case metric > uint32(vec.Dot):
		return nil, corrupt("unknown metric %d", metric)
	case seed == 0:
		return nil, corrupt("zero seed")
	case count > 0 && entrySlot >= count:
		return nil, corrupt("entry point %d outside %d slots", entrySlot, count)
	}

	ix := New(int(dim), Config{
		M:              int(m),
		EfConstruction: int(efc),
		Metric:         vec.Metric(metric),
		Seed:           seed,
	})

	// Chunks are allocated one at a time, each only after the previous one
	// was read in full, so a stream that lies about its length fails on a
	// short read long before memory becomes a problem.
	var tab []*chunk
	rec := ix.recordSize()
	var buf []byte
	for base := 0; base < int(count); base += chunkSize {
		c := ix.newChunk()
		tab = append(tab, c)
		n := min(chunkSize, int(count)-base)

		buf = resize(buf, n*rec)
		if err := cr.read(buf); err != nil {
			return nil, err
		}
		upperWords := 0
		for i := 0; i < n; i++ {
			nd := &c.nodes[i]
			b := buf[i*rec : (i+1)*rec]
			nd.id = le.Uint64(b)
			nd.level = b[8]
			if nd.level > maxLevel || b[9] > 1 {
				return nil, corrupt("slot %d: bad level or tombstone flag", base+i)
			}
			nd.deleted.Store(b[9] == 1)
			b = b[nodeFixedSize:]
			v := c.vecs[i*ix.dim : (i+1)*ix.dim]
			for j := range v {
				v[j] = math.Float32frombits(le.Uint32(b[4*j:]))
			}
			b = b[4*ix.dim:]
			l := c.links[i*ix.stride0 : (i+1)*ix.stride0]
			for j := range l {
				l[j] = le.Uint32(b[4*j:])
			}
			upperWords += int(nd.level) * ix.stride
		}

		buf = resize(buf, 4*upperWords)
		if err := cr.read(buf); err != nil {
			return nil, err
		}
		off := 0
		for i := 0; i < n; i++ {
			nd := &c.nodes[i]
			if nd.level == 0 {
				continue
			}
			nd.upper = make([]uint32, int(nd.level)*ix.stride)
			for j := range nd.upper {
				nd.upper[j] = le.Uint32(buf[off:])
				off += 4
			}
		}
	}

	var trailer [4]byte
	if _, err := io.ReadFull(r, trailer[:]); err != nil {
		return nil, readErr(err)
	}
	if le.Uint32(trailer[:]) != cr.sum {
		return nil, corrupt("checksum mismatch")
	}

	if err := ix.adopt(tab, count, entrySlot); err != nil {
		return nil, err
	}
	return ix, nil
}

// adopt installs a decoded arena after checking every invariant the search
// code relies on to stay in bounds. The checksum already makes random damage
// vanishingly unlikely to get this far; this is what keeps a well-formed but
// wrong file from causing a panic in some later Search.
func (ix *Index) adopt(tab []*chunk, count, entrySlot uint32) error {
	for slot := uint32(0); slot < count; slot++ {
		c := tab[slot>>chunkShift]
		i := slot & chunkMask
		nd := &c.nodes[i]
		for layer := 0; layer <= int(nd.level); layer++ {
			l := ix.links(c, i, layer)
			if int(l[0]) > len(l)-1 {
				return corrupt("slot %d layer %d: %d neighbours", slot, layer, l[0])
			}
			for _, nb := range l[1 : 1+l[0]] {
				// A neighbour on this layer must itself have this layer,
				// or expanding it would index past its upper lists.
				if nb >= count || nb == slot ||
					int(tab[nb>>chunkShift].nodes[nb&chunkMask].level) < layer {
					return corrupt("slot %d layer %d: bad neighbour %d", slot, layer, nb)
				}
			}
		}
		if nd.deleted.Load() {
			continue
		}
		if _, dup := ix.ids[nd.id]; dup {
			return corrupt("id %d is live in two slots", nd.id)
		}
		ix.ids[nd.id] = slot
	}
	ix.count = count
	ix.live = len(ix.ids)
	ix.table.Store(&tab)
	if count > 0 {
		level := tab[entrySlot>>chunkShift].nodes[entrySlot&chunkMask].level
		ix.entry.Store(packEntry(entrySlot, int(level)))
	}
	return nil
}
