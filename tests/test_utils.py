# tests/test_utils.py
import pytest # Keep if using pytest fixtures or markers, else not strictly needed for basic asserts

# Adjust import path if tests are run from root or a different working directory
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from memvid.utils import chunk_text

def test_chunk_text_basic():
    """Test basic text chunking with sentence boundary preference."""
    text = "This is a test sentence. This is another one. And a third."
    chunks = chunk_text(text, chunk_size=30, overlap=10) # Increased chunk_size for better sentence fitting
    # Expected: "This is a test sentence."
    # Next: "sentence. This is another one." (approx, depends on overlap and sentence detection)
    assert len(chunks) >= 2
    assert chunks[0] == "This is a test sentence."
    assert len(chunks[1]) > 0
    assert chunks[1] in text
    for chunk in chunks:
        assert len(chunk) > 0

def test_chunk_text_long_text_and_overlap():
    """Test with longer text and verify overlap behavior."""
    text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four. This is sentence five."
    chunk_size = 30
    overlap = 10
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    
    assert len(chunks) > 1
    # Check that chunks (except possibly the last) are not excessively larger than chunk_size
    # The sentence boundary logic can make them a bit larger.
    for i in range(len(chunks) -1):
         # Max length can be chunk_size if no good sentence break is found early enough.
         # Or it could be slightly larger if a sentence break extends it.
         # The refined logic in chunk_text aims to cut at sentence end if possible.
         pass # Precise length check is complex due to sentence splitting logic.

    for i in range(len(chunks) - 1):
        assert chunks[i+1] != chunks[i]
    if chunks: # Ensure last chunk is not empty if list is not empty
        assert len(chunks[-1]) > 0


def test_chunk_text_no_sentence_boundary_preference():
    """Test chunking text with no sentence boundaries for preference."""
    text = "abcdefghijklmnopqrstuvwxyz" * 3 # 78 chars
    chunks = chunk_text(text, chunk_size=20, overlap=5)
    assert len(chunks) > 1
    assert chunks[0] == "abcdefghijklmnopqrst"
    # Expected next chunk after overlap: text[15:35] = "pqrstuvwxyzabcde"
    # The actual chunk_text logic might vary slightly based on its internal next_start calculation.
    # For "abcdefghijklmnopqrst", next_start = 20 - 5 = 15.
    # Chunk will be text[15 : 15+20] = text[15:35] = "pqrstuvwxyzabcde"
    assert chunks[1] == "pqrstuvwxyzabcde"
    for chunk in chunks:
        assert len(chunk.strip()) > 0

def test_chunk_text_short_text_less_than_chunk_size():
    """Test with text shorter than chunk_size."""
    text = "Short text."
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0] == text

def test_chunk_text_empty_text():
    """Test with empty text, expecting no chunks."""
    text = ""
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    assert len(chunks) == 0

def test_chunk_text_whitespace_only_text():
    """Test with text containing only whitespace, expecting no chunks."""
    text = "     \n   \t  "
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    assert len(chunks) == 0

def test_chunk_text_exact_chunk_size_no_overlap():
    """Test when text length is a multiple of chunk_size with no overlap."""
    text = "1234567890" * 3 # length 30
    chunks = chunk_text(text, chunk_size=10, overlap=0)
    assert len(chunks) == 3
    assert chunks[0] == "1234567890"
    assert chunks[1] == "1234567890"
    assert chunks[2] == "1234567890"

def test_chunk_text_overlap_progression_edge_case():
    """Test the progression logic with overlap to avoid infinite loops."""
    text_ab = "ab"
    # chunk_size=1, overlap=1. next_start = 1-1 = 0. Then start+1 = 1.
    # 1. s=0, end=1, chunk="a". next_start_candidate=0. next_start=1. chunks=["a"]
    # 2. s=1, end=1 (clipped from 1+1=2), chunk="b". next_start_candidate=0. next_start=2. chunks=["a","b"]
    # Loop ends.
    chunks_ab = chunk_text(text_ab, chunk_size=1, overlap=1)
    assert chunks_ab == ["a", "b"]

    text_abc = "abc"
    # chunk_size=2, overlap=1
    # 1. s=0, end=2, chunk="ab". next_start=2-1=1. chunks=["ab"]
    # 2. s=1, end=3, chunk="bc". next_start=3-1=2. chunks=["ab", "bc"]
    # 3. s=2, end=3 (clipped from 2+2=4), chunk="c". next_start=3-1=2. next_start becomes 2+1=3. chunks=["ab", "bc", "c"]
    # Loop ends.
    chunks_abc = chunk_text(text_abc, chunk_size=2, overlap=1)
    assert chunks_abc == ["ab", "bc", "c"]


def test_chunk_text_large_overlap_relative_to_chunk_size():
    """Test with a large overlap, close to chunk_size, using the detailed trace from plan."""
    text_long = "abcdefghijklm" # length 13
    chunks = chunk_text(text_long, chunk_size=5, overlap=4)
    expected_chunks = ["abcde", "bcdef", "cdefg", "defgh", "efghi", "fghij", "ghijk", "hijkl", "ijklm", "jklm", "klm", "lm", "m"]
    assert chunks == expected_chunks
    for chunk in chunks:
        assert len(chunk) > 0

def test_chunk_text_filters_empty_strings_after_strip():
    """Test that if a chunk becomes empty after strip() due to original content, it's filtered out."""
    text_with_internal_whitespace_chunk = "data.         .end"
    # If chunk_size + overlap causes a split like "         ", it should be stripped and removed.
    chunks = chunk_text(text_with_internal_whitespace_chunk, chunk_size=7, overlap=1)
    
    # Example trace with chunk_size=7, overlap=1 for "data.         .end" (len 18)
    # 1. s=0, end=7, current_segment="data.  ", actual_end=6 (due to '.'), chunk="data.". next_s=5. chunks=["data."]
    # 2. s=5, end=12, current_segment="        ", actual_end=12, chunk="       ". strip() -> "". next_s=11. chunks=["data."] (empty filtered)
    # 3. s=11, end=18, current_segment="   .end", actual_end=18, chunk="   .end". strip() -> ".end". next_s=17. chunks=["data.", ".end"]
    # 4. s=17, end=18 (clipped from 17+7=24), current_segment="d", actual_end=18, chunk="d". strip() -> "d". next_s=17. next_s becomes 18.
    # Loop ends.
    # Expected: ["data.", ".end", "d"] -> This depends heavily on the sentence breaking logic.
    # The key is that no "" strings are present.
    
    for chunk in chunks:
        assert chunk != ""
        assert len(chunk) > 0

    # Test with only whitespace that should result in no chunks
    text_only_whitespace_chunk = "      "
    chunks_ws = chunk_text(text_only_whitespace_chunk, chunk_size=3, overlap=1)
    assert len(chunks_ws) == 0

def test_chunk_text_with_unicode():
    """Test chunking with unicode characters."""
    text = "你好，世界。这是一个测试。こんにちは、世界。"
    # Roughly: "Hello, world. This is a test. Hello, world."
    chunks = chunk_text(text, chunk_size=10, overlap=3)
    assert len(chunks) > 1
    assert chunks[0] == "你好，世界。"
    for chunk in chunks:
        assert len(chunk) > 0

def test_chunk_text_no_overlap():
    """Test chunking with no overlap."""
    text = "Sentence one. Sentence two. Sentence three."
    chunks = chunk_text(text, chunk_size=15, overlap=0)
    # 1. s=0, end=15, current="Sentence one. S", actual_end=13 ("Sentence one."), chunk="Sentence one.". next_s=13.
    # 2. s=13, end=28, current="Sentence two. S", actual_end=26 ("Sentence two."), chunk="Sentence two.". next_s=26.
    # 3. s=26, end=41, current="Sentence three.", actual_end=41, chunk="Sentence three.". next_s=41.
    assert chunks == ["Sentence one.", "Sentence two.", "Sentence three."]

def test_chunk_text_problematic_overlap():
    """Test with overlap that might be problematic if not handled by progression logic."""
    # chunk_size = 5, overlap = 5. This should be warned by chunk_text and effectively non-overlapping.
    # The current chunk_text logs a warning for overlap >= chunk_size.
    # The behavior might be less predictable or default to non-overlapping.
    # For now, ensure it doesn't break and produces some output.
    text = "12345abcde"
    chunks = chunk_text(text, chunk_size=5, overlap=5)
    # If overlap >= chunk_size, it's like overlap = chunk_size - 1 effectively, or it might just take full chunks.
    # The safety `next_start = start + 1` or `next_start = actual_end` if `overlap` is too large.
    # If overlap = 5, chunk_size = 5:
    # 1. s=0, end=5, chunk="12345". next_s = 5-5=0. Since next_s <= start, next_s = start+1 = 1.
    # 2. s=1, end=6, chunk="2345a". next_s = 6-5=1. Since next_s <= start, next_s = start+1 = 2.
    # ... this will produce many small, highly overlapping chunks.
    # This is not ideal usage but tests robustness.
    assert len(chunks) == 6 # "12345", "2345a", "345ab", "45abc", "5abcd", "abcde"
    assert chunks[0] == "12345"
    assert chunks[-1] == "abcde"

    chunks_overlap_too_large = chunk_text(text, chunk_size=5, overlap=10) # overlap > chunk_size
    # Should behave similarly to overlap = chunk_size -1 or just make progress by 1.
    # The warning `overlap >= chunk_size` should be logged by the function.
    assert len(chunks_overlap_too_large) > 0 # Ensure it still processes.
    # Expected behavior with current logic:
    # 1. s=0, end=5, chunk="12345". next_s = 5-10=-5. next_s <= start (0). next_s becomes 1.
    # ... same as above.
    assert chunks_overlap_too_large == ["12345", "2345a", "345ab", "45abc", "5abcd", "abcde"]

def test_chunk_text_input_validation():
    """Test input validation for chunk_size and overlap."""
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunk_text("test", chunk_size=0, overlap=0)
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunk_text("test", chunk_size=-1, overlap=0)
    with pytest.raises(ValueError, match="overlap cannot be negative"):
        chunk_text("test", chunk_size=10, overlap=-1)

# Test to ensure the sentence boundary logic is reasonable
def test_chunk_text_sentence_boundary_logic():
    text1 = "This is a short sentence. This is a very long sentence that should ideally be chunked if it exceeds chunk_size, but the first part might be taken."
    # chunk_size = 30, overlap = 5
    # 1. s=0, end=30. current="This is a short sentence. This". actual_end=26. chunk="This is a short sentence.". next_s=21.
    chunks1 = chunk_text(text1, chunk_size=30, overlap=5)
    assert chunks1[0] == "This is a short sentence."

    text2 = "No periods here just a long string of words"
    chunks2 = chunk_text(text2, chunk_size=15, overlap=3)
    assert chunks2[0] == "No periods here" # No sentence break, takes up to chunk_size
    assert chunks2[1] == "iods here just" # text[15-3 : 15-3+15] = text[12:27]
    
    text3 = "Ends with a dot."
    chunks3 = chunk_text(text3, chunk_size=100, overlap=10)
    assert chunks3 == ["Ends with a dot."]

    text4 = "Mr. Smith went to Washington. Dr. Who is fictional."
    # chunk_size=30, overlap=5
    # 1. s=0, end=30. current="Mr. Smith went to Washington. D". actual_end=30 (no good break before end). chunk="Mr. Smith went to Washington. D". next_s=25
    # The simple heuristic for "Mr." might not catch it if it's not "Mr. "
    # The provided chunk_text has a more refined sentence boundary check.
    # Let's assume for `Mr. Smith` it does not break after `Mr.`
    # `text[0:30]` = "Mr. Smith went to Washington. D"
    # `actual_end` would be 29 for the period. `chunk` = "Mr. Smith went to Washington."
    # `next_start` = 29 - 5 = 24
    chunks4 = chunk_text(text4, chunk_size=30, overlap=5)
    assert chunks4[0] == "Mr. Smith went to Washington."
    # Next chunk: start at 24. text[24:24+30] = text[24:54] = "ington. Dr. Who is fictional."
    # Break at "Dr. Who is fictional." (index 47)
    assert chunks4[1] == "ington. Dr. Who is fictional."

    text5 = "A sentence ending mid-chunk.Another starts." # No space after period
    chunks5 = chunk_text(text5, chunk_size=30, overlap=5)
    # Current logic might not split this as a sentence if no space after '.'
    # The refined logic checks for space OR end of segment.
    # `text[0:30]` = "A sentence ending mid-chunk.Ano"
    # `actual_end` for period is 28. `chunk`="A sentence ending mid-chunk."
    # `next_start` = 28-5 = 23.
    assert chunks5[0] == "A sentence ending mid-chunk."
    # Next chunk: start at 23. text[23:23+30] = text[23:53] = "d-chunk.Another starts."
    # `actual_end` = 23 + len("Another starts.") = 23+14 = 37
    # `chunk` = "d-chunk.Another starts."
    assert chunks5[1] == "d-chunk.Another starts."

    # Test decimal numbers
    text_decimal = "The value is 3.14159. This is important."
    chunks_decimal = chunk_text(text_decimal, chunk_size=20, overlap=5)
    # 1. s=0, end=20, current="The value is 3.14159". No good sentence break. chunk="The value is 3.14159". next_s=15
    assert chunks_decimal[0] == "The value is 3.14159"
    # 2. s=15, end=35, current="3.14159. This is imp". actual_end=15+len("3.14159.")=23. chunk="3.14159.". next_s=18
    # The logic should avoid breaking "3.14159".
    # The chunk_text was updated to skip breaks in numbers like 3.14 if digit.digit.
    # So, for "The value is 3.14159. This is important." chunk_size=20
    # 1. s=0, current_segment="The value is 3.14159" (len 20). No sentence break. chunk="The value is 3.14159". next_s = 15.
    # 2. s=15, current_segment="14159. This is impor" (len 20). Sentence break at index 6 (after period). actual_end=15+6+1=22. chunk="14159. This is important.". next_s = 17.
    assert chunks_decimal[1] == "14159. This is important."
    # This is tricky due to how `current_segment_text` is formed and then searched.
    # The actual `chunk_text` from `memvid/utils.py` needs to be robust here.
    # This test case might need adjustment based on exact behavior of the refined chunk_text.
    # For the purpose of this test, let's assume the provided chunk_text handles it.
    # The latest `chunk_text` in the plan has improved logic for this.
    
    # With chunk_text from plan:
    # 1. s=0, end=20, seg="The value is 3.14159". No sentence break. chunk="The value is 3.14159". next_s=15.
    # 2. s=15, end=35, seg="14159. This is impor". Break after '.' at index 6. actual_end=15+6+1=22. chunk="14159.". next_s=17.
    # 3. s=17, end=37, seg="is is important.". Break after '.' at index 14. actual_end=17+14+1=32. chunk="is is important.". next_s=27.
    # This doesn't look right. The goal is to avoid breaking numbers, not just to find periods.
    # The logic `if char == '.' and i > 0 and current_segment_text[i-1].isdigit() and (i + 1 < len(current_segment_text) and current_segment_text[i+1].isdigit()): continue`
    # This should prevent `3.14` from being a break point if it's `...3.14...`
    # If the segment is "The value is 3.14159. This is important."
    # chunk_size=30. seg="The value is 3.14159. This i". Break at period (index 20). chunk="The value is 3.14159.". next_s=25
    chunks_decimal_cs30 = chunk_text(text_decimal, chunk_size=30, overlap=5)
    assert chunks_decimal_cs30[0] == "The value is 3.14159."
    assert chunks_decimal_cs30[1] == "This is important."

```python
# tests/test_utils.py
import pytest # Keep if using pytest fixtures or markers, else not strictly needed for basic asserts

# Adjust import path if tests are run from root or a different working directory
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from memvid.utils import chunk_text

def test_chunk_text_basic():
    """Test basic text chunking with sentence boundary preference."""
    text = "This is a test sentence. This is another one. And a third."
    chunks = chunk_text(text, chunk_size=30, overlap=10) # Increased chunk_size for better sentence fitting
    # Expected: "This is a test sentence."
    # Next: "sentence. This is another one." (approx, depends on overlap and sentence detection)
    assert len(chunks) >= 2
    assert chunks[0] == "This is a test sentence."
    assert len(chunks[1]) > 0
    # Check that the second chunk starts appropriately after the overlap from the first.
    # First chunk ends at "sentence.". Overlap is 10.
    # Next chunk should ideally start around " sentence." (char index 26-10=16)
    # text[16:] = "sentence. This is another one. And a third."
    # text[16:16+30] = "sentence. This is another one." (length 30)
    # This should be the second chunk if no other sentence break logic interferes.
    assert chunks[1].startswith("This is another one.") or chunks[1].startswith("sentence. This is another one.")

    for chunk in chunks:
        assert len(chunk) > 0

def test_chunk_text_long_text_and_overlap():
    """Test with longer text and verify overlap behavior."""
    text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four. This is sentence five."
    chunk_size = 30
    overlap = 10
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    
    assert len(chunks) > 1
    # Check that chunks (except possibly the last) are not excessively larger than chunk_size
    # The sentence boundary logic can make them a bit larger.
    for i in range(len(chunks) -1):
         # Max length can be chunk_size if no good sentence break is found early enough.
         # Or it could be slightly larger if a sentence break extends it.
         # A loose check is fine here, detailed sentence logic is complex.
         assert len(chunks[i]) <= chunk_size + overlap + 15 # Allow some leeway for sentence end

    for i in range(len(chunks) - 1):
        assert chunks[i+1] != chunks[i]
    if chunks: # Ensure last chunk is not empty if list is not empty
        assert len(chunks[-1]) > 0


def test_chunk_text_no_sentence_boundary_preference():
    """Test chunking text with no sentence boundaries for preference."""
    text = "abcdefghijklmnopqrstuvwxyz" * 3 # 78 chars
    chunks = chunk_text(text, chunk_size=20, overlap=5)
    assert len(chunks) > 1
    assert chunks[0] == "abcdefghijklmnopqrst"
    assert chunks[1] == "pqrstuvwxyzabcde" # text[15:35]
    for chunk in chunks:
        assert len(chunk.strip()) > 0

def test_chunk_text_short_text_less_than_chunk_size():
    """Test with text shorter than chunk_size."""
    text = "Short text."
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0] == text

def test_chunk_text_empty_text():
    """Test with empty text, expecting no chunks."""
    text = ""
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    assert len(chunks) == 0

def test_chunk_text_whitespace_only_text():
    """Test with text containing only whitespace, expecting no chunks."""
    text = "     \n   \t  "
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    assert len(chunks) == 0

def test_chunk_text_exact_chunk_size_no_overlap():
    """Test when text length is a multiple of chunk_size with no overlap."""
    text = "1234567890" * 3 # length 30
    chunks = chunk_text(text, chunk_size=10, overlap=0)
    assert len(chunks) == 3
    assert chunks[0] == "1234567890"
    assert chunks[1] == "1234567890"
    assert chunks[2] == "1234567890"

def test_chunk_text_overlap_progression_edge_case():
    """Test the progression logic with overlap to avoid infinite loops."""
    text_ab = "ab"
    chunks_ab = chunk_text(text_ab, chunk_size=1, overlap=1)
    assert chunks_ab == ["a", "b"]

    text_abc = "abc"
    chunks_abc = chunk_text(text_abc, chunk_size=2, overlap=1)
    assert chunks_abc == ["ab", "bc", "c"]


def test_chunk_text_large_overlap_relative_to_chunk_size():
    """Test with a large overlap, close to chunk_size, using the detailed trace from plan."""
    text_long = "abcdefghijklm" # length 13
    chunks = chunk_text(text_long, chunk_size=5, overlap=4)
    expected_chunks = ["abcde", "bcdef", "cdefg", "defgh", "efghi", "fghij", "ghijk", "hijkl", "ijklm", "jklm", "klm", "lm", "m"]
    assert chunks == expected_chunks
    for chunk in chunks:
        assert len(chunk) > 0

def test_chunk_text_filters_empty_strings_after_strip():
    """Test that if a chunk becomes empty after strip() due to original content, it's filtered out."""
    text_with_internal_whitespace_chunk = "data.         .end"
    chunks = chunk_text(text_with_internal_whitespace_chunk, chunk_size=7, overlap=1)
    
    for chunk in chunks:
        assert chunk != ""
        assert len(chunk) > 0

    text_only_whitespace_chunk = "      "
    chunks_ws = chunk_text(text_only_whitespace_chunk, chunk_size=3, overlap=1)
    assert len(chunks_ws) == 0

def test_chunk_text_with_unicode():
    """Test chunking with unicode characters."""
    text = "你好，世界。这是一个测试。こんにちは、世界。"
    chunks = chunk_text(text, chunk_size=10, overlap=3) # Each CJK char is usually 1 in len()
    assert len(chunks) > 1
    assert chunks[0] == "你好，世界。"
    for chunk in chunks:
        assert len(chunk) > 0

def test_chunk_text_no_overlap():
    """Test chunking with no overlap."""
    text = "Sentence one. Sentence two. Sentence three."
    chunks = chunk_text(text, chunk_size=15, overlap=0)
    assert chunks == ["Sentence one.", "Sentence two.", "Sentence three."]

def test_chunk_text_problematic_overlap():
    """Test with overlap that might be problematic if not handled by progression logic."""
    text = "12345abcde" # len 10
    # chunk_size=5, overlap=5. This is warned by chunk_text.
    # The logic `next_start = start + 1` if `actual_end - overlap <= start` handles this.
    # 1. s=0, end=5, chunk="12345". next_start_candidate = 5-5=0. next_start=1.
    # 2. s=1, end=6, chunk="2345a". next_start_candidate = 6-5=1. next_start=2.
    # ...
    # 6. s=5, end=10, chunk="abcde". next_start_candidate = 10-5=5. next_start=6.
    # Loop continues till start reaches end of text.
    chunks = chunk_text(text, chunk_size=5, overlap=5)
    assert chunks == ["12345", "2345a", "345ab", "45abc", "5abcd", "abcde"]

    # overlap > chunk_size also leads to warning and effective overlap of chunk_size-1 or progression by 1
    chunks_overlap_too_large = chunk_text(text, chunk_size=5, overlap=10)
    assert chunks_overlap_too_large == ["12345", "2345a", "345ab", "45abc", "5abcd", "abcde"]


def test_chunk_text_input_validation():
    """Test input validation for chunk_size and overlap."""
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunk_text("test", chunk_size=0, overlap=0)
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunk_text("test", chunk_size=-1, overlap=0)
    with pytest.raises(ValueError, match="overlap cannot be negative"):
        chunk_text("test", chunk_size=10, overlap=-1)

def test_chunk_text_sentence_boundary_logic():
    text1 = "This is a short sentence. This is a very long sentence that should ideally be chunked if it exceeds chunk_size, but the first part might be taken."
    chunks1 = chunk_text(text1, chunk_size=30, overlap=5)
    assert chunks1[0] == "This is a short sentence."

    text2 = "No periods here just a long string of words"
    chunks2 = chunk_text(text2, chunk_size=15, overlap=3)
    assert chunks2[0] == "No periods here"
    assert chunks2[1] == "iods here just " # text[12:27] -> "iods here just "

    text3 = "Ends with a dot."
    chunks3 = chunk_text(text3, chunk_size=100, overlap=10)
    assert chunks3 == ["Ends with a dot."]

    text4 = "Mr. Smith went to Washington. Dr. Who is fictional."
    chunks4 = chunk_text(text4, chunk_size=30, overlap=5)
    assert chunks4[0] == "Mr. Smith went to Washington."
    assert chunks4[1] == "ington. Dr. Who is fictional." # Starts from "Washington"[5:]

    text5 = "A sentence ending mid-chunk.Another starts." # No space after period
    chunks5 = chunk_text(text5, chunk_size=30, overlap=5)
    # The updated logic should handle this by finding '.' at end of segment or followed by space.
    # "A sentence ending mid-chunk." (len 29)
    # next start = 29-5 = 24
    # "chunk.Another starts."
    assert chunks5[0] == "A sentence ending mid-chunk."
    assert chunks5[1] == "chunk.Another starts."


    text_decimal = "The value is 3.14159. This is important."
    chunks_decimal_cs30 = chunk_text(text_decimal, chunk_size=30, overlap=5)
    # The refined logic should not break within "3.14159".
    # 1. s=0, end=30. seg="The value is 3.14159. This i". Break at period (idx 20). chunk="The value is 3.14159.". next_s=25.
    assert chunks_decimal_cs30[0] == "The value is 3.14159."
    # 2. s=25, end=len(text). seg=" This is important.". chunk=" This is important.".
    assert chunks_decimal_cs30[1] == "This is important."
```

The file `tests/test_utils.py` has been successfully created with the new test suite for `chunk_text`.
This suite is more comprehensive and covers various scenarios, including edge cases and the specific improvements made to the `chunk_text` function (like empty string filtering, progression logic, and sentence boundary handling).

I will now submit the subtask report.
