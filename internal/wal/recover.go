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
)

// Recovery has to tell two things apart that look the same at the point of
// failure, a record that does not verify:
//
//   - a torn tail: the process or machine died while the last records were
//     being written. Nothing after the tear was ever acknowledged as
//     durable, so cutting it off loses nothing the caller was promised.
//   - corruption: bytes that were once good have changed. Cutting the log
//     there would silently drop acknowledged records, so Open must refuse.
//
// The rule:
//
//  1. Sealed segments (every segment but the last) are fsynced in full
//     before their successor is created, under every sync policy. They have
//     no tail to tear, so any bad record in one is corruption.
//  2. In the last segment, a bad record at offset X is a torn tail only if
//     no valid record starts at any later offset in the file. If one does,
//     good data was written after X, so X was once good too: corruption.
//
// Rule 2 scans every byte offset after X instead of trusting the bad
// record's own length field to find the next boundary. That field is part
// of what may be damaged, and following a wrong length would step over the
// real records behind it and misread corruption as a tear.
//
// The rule errs towards refusing to open. A single damaged record that
// happens to be the very last one is indistinguishable from a tear and is
// truncated; that is inherent to any log without a separate commit marker.

// scanSealed validates a sealed segment and returns how many records it holds.
func scanSealed(path string) (uint64, error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return 0, err
	}
	if err := readHeader(f); err != nil {
		return 0, fmt.Errorf("%s: %w", filepath.Base(path), err)
	}
	r := newRecordReader(f, st.Size())
	var count uint64
	for {
		_, err := r.next(false)
		if err == io.EOF {
			return count, nil
		}
		if err == errBadRecord {
			return 0, fmt.Errorf("%w: sealed segment %s, offset %d", ErrCorrupt, filepath.Base(path), r.offset)
		}
		if err != nil {
			return 0, err
		}
		count++
	}
}

// recoverActive validates the last segment, repairs a torn tail in place,
// and returns the file open for appending along with its size and record
// count.
func recoverActive(path string) (f *os.File, size int64, count uint64, err error) {
	f, err = os.OpenFile(path, os.O_RDWR, 0)
	if err != nil {
		return nil, 0, 0, err
	}
	defer func() {
		if err != nil {
			f.Close()
			f = nil
		}
	}()
	st, err := f.Stat()
	if err != nil {
		return nil, 0, 0, err
	}
	size = st.Size()

	h := make([]byte, headerSize)
	n, rerr := f.ReadAt(h, 0)
	if rerr != nil && !errors.Is(rerr, io.EOF) {
		return nil, 0, 0, rerr
	}
	if size < headerSize || (size == headerSize && string(h[:4]) != segmentMagic) {
		// Crash while the segment was being created. createSegment syncs
		// the header before any record is written, so a file with no room
		// for a record and no good header never held one and can be
		// re-initialised.
		if err = f.Truncate(0); err != nil {
			return nil, 0, 0, err
		}
		if _, err = f.WriteAt(segmentHeader(), 0); err != nil {
			return nil, 0, 0, err
		}
		size = headerSize
	} else if err = checkHeader(h[:n]); err != nil {
		return nil, 0, 0, fmt.Errorf("%s: %w", filepath.Base(path), err)
	}

	r := newRecordReader(f, size)
	for {
		_, err = r.next(false)
		if err == io.EOF {
			break
		}
		if err == errBadRecord {
			var later bool
			if later, err = hasRecordAfter(f, r.offset, size); err != nil {
				return nil, 0, 0, err
			}
			if later {
				err = fmt.Errorf("%w: segment %s, offset %d, valid records follow", ErrCorrupt, filepath.Base(path), r.offset)
				return nil, 0, 0, err
			}
			if err = f.Truncate(r.offset); err != nil {
				return nil, 0, 0, err
			}
			size = r.offset
			break
		}
		if err != nil {
			return nil, 0, 0, err
		}
		count++
	}

	// Makes the truncation durable, and also covers records that survived a
	// process crash only in the page cache: once Open returns, everything
	// LastLSN reports is on disk.
	if err = f.Sync(); err != nil {
		return nil, 0, 0, err
	}
	return f, size, count, nil
}

// hasRecordAfter reports whether a complete, checksummed record starts at
// any offset in (bad, size). A false positive needs a 32-bit CRC to match by
// chance, or a torn payload that itself contains WAL-framed bytes; both make
// Open refuse rather than lose data.
//
// Cost is one pass over the tail plus a CRC for each offset whose length
// field happens to fit in the file, so it is linear on zeros and typical
// data and quadratic only on adversarial content. It runs only when
// recovery has already hit a bad record.
func hasRecordAfter(f *os.File, bad, size int64) (bool, error) {
	start := bad + 1
	if size-start < frameHeaderSize {
		return false, nil
	}
	br := bufio.NewReaderSize(io.NewSectionReader(f, start, size-start), 256<<10)
	var h [frameHeaderSize]byte
	if _, err := io.ReadFull(br, h[:]); err != nil {
		return false, err
	}
	buf := make([]byte, 64<<10)
	for off := start; ; off++ {
		n := int64(binary.LittleEndian.Uint32(h[0:4]))
		if n <= MaxRecordSize && off+frameHeaderSize+n <= size {
			sum := crc32.Checksum(h[0:4], castagnoli)
			body := io.NewSectionReader(f, off+frameHeaderSize, n)
			for {
				k, err := body.Read(buf)
				sum = crc32.Update(sum, castagnoli, buf[:k])
				if err == io.EOF {
					break
				}
				if err != nil {
					return false, err
				}
			}
			if sum == binary.LittleEndian.Uint32(h[4:8]) {
				return true, nil
			}
		}
		b, err := br.ReadByte()
		if err == io.EOF {
			return false, nil
		}
		if err != nil {
			return false, err
		}
		copy(h[:], h[1:])
		h[frameHeaderSize-1] = b
	}
}
