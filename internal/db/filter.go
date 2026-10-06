package db

import (
	"fmt"
	"strings"
)

// Filter restricts a search to points whose payload matches. Every Must
// condition has to hold, no MustNot condition may hold, and if Should is
// non-empty at least one of its conditions has to hold.
type Filter struct {
	Must    []Condition `json:"must,omitempty"`
	Should  []Condition `json:"should,omitempty"`
	MustNot []Condition `json:"must_not,omitempty"`
}

// Condition tests one payload field. Key may be a dotted path into nested
// objects. Op is one of eq, ne, in, lt, lte, gt, gte, exists.
type Condition struct {
	Key   string `json:"key"`
	Op    string `json:"op"`
	Value any    `json:"value,omitempty"`
}

// Validate rejects unknown operators and operand types up front, so Matches
// never has to report an error from inside the index traversal.
func (f *Filter) Validate() error {
	if f == nil {
		return nil
	}
	for _, group := range [][]Condition{f.Must, f.Should, f.MustNot} {
		for _, c := range group {
			if err := c.validate(); err != nil {
				return err
			}
		}
	}
	return nil
}

func (c Condition) validate() error {
	if c.Key == "" {
		return fmt.Errorf("%w: filter condition has no key", ErrInvalid)
	}
	switch c.Op {
	case "eq", "ne":
		if !isScalar(c.Value) {
			return fmt.Errorf("%w: filter %q on %q needs a string, number or bool", ErrInvalid, c.Op, c.Key)
		}
	case "in":
		list, ok := c.Value.([]any)
		if !ok || len(list) == 0 {
			return fmt.Errorf("%w: filter \"in\" on %q needs a non-empty array", ErrInvalid, c.Key)
		}
		for _, v := range list {
			if !isScalar(v) {
				return fmt.Errorf("%w: filter \"in\" on %q has a non-scalar element", ErrInvalid, c.Key)
			}
		}
	case "lt", "lte", "gt", "gte":
		switch c.Value.(type) {
		case float64, string:
		default:
			return fmt.Errorf("%w: filter %q on %q needs a number or string", ErrInvalid, c.Op, c.Key)
		}
	case "exists":
		if _, ok := c.Value.(bool); !ok && c.Value != nil {
			return fmt.Errorf("%w: filter \"exists\" on %q takes true or false", ErrInvalid, c.Key)
		}
	default:
		return fmt.Errorf("%w: unknown filter op %q", ErrInvalid, c.Op)
	}
	return nil
}

func isScalar(v any) bool {
	switch v.(type) {
	case float64, string, bool:
		return true
	}
	return false
}

// Matches reports whether payload satisfies the filter. A nil filter matches
// everything; a nil payload has no fields.
func (f *Filter) Matches(payload map[string]any) bool {
	if f == nil {
		return true
	}
	for _, c := range f.Must {
		if !c.matches(payload) {
			return false
		}
	}
	for _, c := range f.MustNot {
		if c.matches(payload) {
			return false
		}
	}
	if len(f.Should) == 0 {
		return true
	}
	for _, c := range f.Should {
		if c.matches(payload) {
			return true
		}
	}
	return false
}

func (c Condition) matches(payload map[string]any) bool {
	got, ok := lookup(payload, c.Key)
	switch c.Op {
	case "exists":
		want, isBool := c.Value.(bool)
		return ok == (want || !isBool)
	case "ne":
		return !ok || got != c.Value
	}
	if !ok {
		return false
	}
	switch c.Op {
	case "eq":
		return got == c.Value
	case "in":
		for _, v := range c.Value.([]any) {
			if got == v {
				return true
			}
		}
		return false
	}
	cmp, comparable := compare(got, c.Value)
	if !comparable {
		return false
	}
	switch c.Op {
	case "lt":
		return cmp < 0
	case "lte":
		return cmp <= 0
	case "gt":
		return cmp > 0
	default:
		return cmp >= 0
	}
}

func lookup(payload map[string]any, key string) (any, bool) {
	if v, ok := payload[key]; ok {
		return v, true
	}
	head, rest, nested := strings.Cut(key, ".")
	if !nested {
		return nil, false
	}
	child, ok := payload[head].(map[string]any)
	if !ok {
		return nil, false
	}
	return lookup(child, rest)
}

// compare orders two values of the same kind. Mixed kinds do not compare.
func compare(a, b any) (int, bool) {
	switch x := a.(type) {
	case float64:
		y, ok := b.(float64)
		if !ok {
			return 0, false
		}
		switch {
		case x < y:
			return -1, true
		case x > y:
			return 1, true
		}
		return 0, true
	case string:
		y, ok := b.(string)
		if !ok {
			return 0, false
		}
		return strings.Compare(x, y), true
	}
	return 0, false
}
