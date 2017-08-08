package tf

import "testing"

func TestExtractNodeName(t *testing.T) {
	cases := []struct {
		in, want string
	}{
		{"", ""},
		{"asdf", "asdf"},
		{"test/foo/bar:10", "test/foo/bar"},
	}

	for i, c := range cases {
		out := ExtractNodeName(c.in)
		if out != c.want {
			t.Errorf("%d. ExtractNodeName(%q) = %q; not %q", i, c.in, out, c.want)
		}
	}
}
