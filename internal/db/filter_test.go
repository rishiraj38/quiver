package db

import (
	"encoding/json"
	"errors"
	"testing"
)

func TestFilterMatches(t *testing.T) {
	var payload map[string]any
	raw := `{"brand":"acme","price":40,"in_stock":true,"tags":["a"],"dims":{"w":3}}`
	if err := json.Unmarshal([]byte(raw), &payload); err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		filter string
		want   bool
	}{
		{`{}`, true},
		{`{"must":[{"key":"brand","op":"eq","value":"acme"}]}`, true},
		{`{"must":[{"key":"brand","op":"eq","value":"other"}]}`, false},
		{`{"must":[{"key":"brand","op":"ne","value":"other"}]}`, true},
		{`{"must":[{"key":"missing","op":"ne","value":"x"}]}`, true},
		{`{"must":[{"key":"missing","op":"eq","value":"x"}]}`, false},
		{`{"must":[{"key":"price","op":"lt","value":50},{"key":"price","op":"gte","value":40}]}`, true},
		{`{"must":[{"key":"price","op":"gt","value":40}]}`, false},
		{`{"must":[{"key":"price","op":"lt","value":"50"}]}`, false},
		{`{"must":[{"key":"brand","op":"in","value":["x","acme"]}]}`, true},
		{`{"must":[{"key":"in_stock","op":"eq","value":true}]}`, true},
		{`{"must":[{"key":"dims.w","op":"eq","value":3}]}`, true},
		{`{"must":[{"key":"dims.h","op":"exists"}]}`, false},
		{`{"must":[{"key":"dims.h","op":"exists","value":false}]}`, true},
		{`{"must":[{"key":"tags","op":"eq","value":"a"}]}`, false},
		{`{"must_not":[{"key":"brand","op":"eq","value":"acme"}]}`, false},
		{`{"should":[{"key":"brand","op":"eq","value":"x"},{"key":"price","op":"lte","value":40}]}`, true},
		{`{"should":[{"key":"brand","op":"eq","value":"x"}]}`, false},
	}
	for _, tc := range cases {
		var f Filter
		if err := json.Unmarshal([]byte(tc.filter), &f); err != nil {
			t.Fatalf("%s: %v", tc.filter, err)
		}
		if err := f.Validate(); err != nil {
			t.Fatalf("%s: %v", tc.filter, err)
		}
		if got := f.Matches(payload); got != tc.want {
			t.Errorf("%s: got %v, want %v", tc.filter, got, tc.want)
		}
	}
	var none *Filter
	if !none.Matches(nil) {
		t.Error("nil filter should match everything")
	}
}

func TestFilterValidate(t *testing.T) {
	bad := []string{
		`{"must":[{"key":"","op":"eq","value":1}]}`,
		`{"must":[{"key":"a","op":"like","value":1}]}`,
		`{"must":[{"key":"a","op":"eq","value":[1]}]}`,
		`{"must":[{"key":"a","op":"in","value":[]}]}`,
		`{"must":[{"key":"a","op":"in","value":3}]}`,
		`{"must":[{"key":"a","op":"lt","value":true}]}`,
		`{"must_not":[{"key":"a","op":"exists","value":"yes"}]}`,
	}
	for _, s := range bad {
		var f Filter
		if err := json.Unmarshal([]byte(s), &f); err != nil {
			t.Fatalf("%s: %v", s, err)
		}
		if err := f.Validate(); !errors.Is(err, ErrInvalid) {
			t.Errorf("%s: got %v, want ErrInvalid", s, err)
		}
	}
}
