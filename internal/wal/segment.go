package wal

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
)

// On-disk layout, all integers little-endian.
//
//	segment file  <first LSN, 20 decimal digits>.wal
//	header        magic "QWAL" (4) | version uint32 (4)
//	record        length uint32 (4) | crc32c uint32 (4) | payload (length bytes)
//
// The checksum covers the length field as well as the payload, so a record
// whose length was damaged fails verification instead of silently framing
// the wrong bytes. It also means a run of zeros never parses as a record:
// the CRC of four zero bytes is not zero.
//
// LSNs are not stored. A record's LSN is the segment's first LSN plus its
// index in the segment, which is what keeps them dense by construction.
const (
	segmentMagic    = "QWAL"
	segmentVersion  = 1
	segmentSuffix   = ".wal"
	headerSize      = 8
	frameHeaderSize = 8

	// MaxRecordSize bounds a single payload. Recovery trusts no length field
	// above it, so a corrupt length cannot ask for a multi-gigabyte buffer.
	MaxRecordSize = 64 << 20
)

var castagnoli = crc32.MakeTable(crc32.Castagnoli)

// errBadRecord means the bytes at the reader's offset are not a complete,
// checksummed record. Whether that is a torn tail or corruption depends on
// where it happened, so the caller decides.
var errBadRecord = errors.New("wal: bad record")

func segmentName(first uint64) string {
	return fmt.Sprintf("%020d%s", first, segmentSuffix)
}

func parseSegmentName(name string) (uint64, bool) {
	digits, ok := strings.CutSuffix(name, segmentSuffix)
	if !ok || len(digits) != 20 {
		return 0, false
	}
	first, err := strconv.ParseUint(digits, 10, 64)
	if err != nil || first == 0 {
		return 0, false
	}
	return first, true
}

// listSegments returns the first LSN of every segment in dir, ascending.
// Anything that is not named like a segment is ignored.
func listSegments(dir string) ([]uint64, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var firsts []uint64
	for _, e := range entries {
		if first, ok := parseSegmentName(e.Name()); ok && !e.IsDir() {
			firsts = append(firsts, first)
		}
	}
	slices.Sort(firsts)
	return firsts, nil
}

func segmentHeader() []byte {
	h := make([]byte, headerSize)
	copy(h, segmentMagic)
	binary.LittleEndian.PutUint32(h[4:], segmentVersion)
	return h
}

func checkHeader(h []byte) error {
	if len(h) < headerSize || string(h[:4]) != segmentMagic {
		return fmt.Errorf("%w: bad segment header", ErrCorrupt)
	}
	if v := binary.LittleEndian.Uint32(h[4:]); v != segmentVersion {
		return fmt.Errorf("wal: unsupported segment version %d", v)
	}
	return nil
}

func readHeader(f *os.File) error {
	h := make([]byte, headerSize)
	if _, err := f.ReadAt(h, 0); err != nil {
		if errors.Is(err, io.EOF) {
			return fmt.Errorf("%w: short segment header", ErrCorrupt)
		}
		return err
	}
	return checkHeader(h)
}

// createSegment creates a segment and makes both its header and its
// directory entry durable before returning. Records are only ever written
// to a segment after this, which recovery relies on: a last segment with a
// damaged header cannot contain acknowledged records.
func createSegment(dir string, first uint64) (*os.File, error) {
	path := filepath.Join(dir, segmentName(first))
	f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		return nil, err
	}
	if _, err = f.WriteAt(segmentHeader(), 0); err == nil {
		if err = f.Sync(); err == nil {
			err = syncDir(dir)
		}
	}
	if err != nil {
		f.Close()
		os.Remove(path)
		return nil, err
	}
	return f, nil
}

// syncDir fsyncs the directory itself. Creating or unlinking a file changes
// the directory, and without this a crash can forget the change even though
// the file's own contents were synced.
func syncDir(dir string) error {
	d, err := os.Open(dir)
	if err != nil {
		return err
	}
	err = d.Sync()
	if cerr := d.Close(); err == nil {
		err = cerr
	}
	return err
}

func checksum(length []byte, payload []byte) uint32 {
	return crc32.Update(crc32.Checksum(length, castagnoli), castagnoli, payload)
}

// appendFrame appends the framed record to dst.
func appendFrame(dst, payload []byte) []byte {
	var h [frameHeaderSize]byte
	binary.LittleEndian.PutUint32(h[0:4], uint32(len(payload)))
	binary.LittleEndian.PutUint32(h[4:8], checksum(h[0:4], payload))
	dst = append(dst, h[:]...)
	return append(dst, payload...)
}

// recordReader streams the records of one segment, holding one record in
// memory at a time.
type recordReader struct {
	br        *bufio.Reader
	offset    int64 // file offset of the next record
	remaining int64 // bytes between offset and the end of the segment
	payload   []byte
}

// newRecordReader reads the records in f[headerSize:size). Passing the size
// in, rather than reading to EOF, lets Replay stop at the point the log had
// reached when it started while appends keep extending the file.
func newRecordReader(f *os.File, size int64) *recordReader {
	n := size - headerSize
	return &recordReader{
		br:        bufio.NewReaderSize(io.NewSectionReader(f, headerSize, n), 256<<10),
		offset:    headerSize,
		remaining: n,
	}
}

// next returns the next record's payload, valid until the following call.
// It returns io.EOF at a clean end and errBadRecord, without advancing, if
// what is there is not a whole valid record. With skip set the payload is
// stepped over unread and unverified.
func (r *recordReader) next(skip bool) ([]byte, error) {
	if r.remaining == 0 {
		return nil, io.EOF
	}
	if r.remaining < frameHeaderSize {
		return nil, errBadRecord
	}
	var h [frameHeaderSize]byte
	if _, err := io.ReadFull(r.br, h[:]); err != nil {
		return nil, readErr(err)
	}
	n := int64(binary.LittleEndian.Uint32(h[0:4]))
	// The second test is the allocation guard that matters in practice: a
	// record cannot be longer than the bytes left in its file.
	if n > MaxRecordSize || n > r.remaining-frameHeaderSize {
		return nil, errBadRecord
	}
	if skip {
		if _, err := r.br.Discard(int(n)); err != nil {
			return nil, readErr(err)
		}
	} else {
		if int64(cap(r.payload)) < n {
			r.payload = make([]byte, n)
		}
		r.payload = r.payload[:n]
		if _, err := io.ReadFull(r.br, r.payload); err != nil {
			return nil, readErr(err)
		}
		if checksum(h[0:4], r.payload) != binary.LittleEndian.Uint32(h[4:8]) {
			return nil, errBadRecord
		}
	}
	r.offset += frameHeaderSize + n
	r.remaining -= frameHeaderSize + n
	if skip {
		return nil, nil
	}
	return r.payload, nil
}

// readErr maps a short read to errBadRecord: the file is shorter than the
// size we were told, which is damage, not an I/O failure.
func readErr(err error) error {
	if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
		return errBadRecord
	}
	return err
}
