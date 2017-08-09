package tf

import "testing"

func TestParseNodeOutput(t *testing.T) {
	cases := []struct {
		in, want string
		wantn    int
	}{
		{"", "", -1},
		{"asdf", "asdf", -1},
		{"test/foo/bar:10", "test/foo/bar", 10},
	}

	for i, c := range cases {
		out, n, err := ParseNodeOutput(c.in)
		if err != nil {
			t.Error(err)
		}
		if out != c.want || n != c.wantn {
			t.Errorf("%d. ExtractNodeName(%q) = %q, %d; not %q, %d", i, c.in, out, n, c.want, c.wantn)
		}
	}
}
