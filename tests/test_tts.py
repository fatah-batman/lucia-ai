"""tests/test_tts.py - Unit tests for Text-to-Speech and sentence boundary splitting."""

from unittest.mock import MagicMock, patch
import pytest

from tts import SENTENCE_BOUNDARY_REGEX, SentenceSpeaker, speak


def test_sentence_boundary_regex_preserves_decimals():
    """Verify regex does not split on decimal numbers like 23.5°C or 0.1 mm."""
    text = (
        "The weather in Bengaluru is 21.2°C (feels like 25.0°C), humidity 97%, "
        "wind 2.6 km/h, precipitation 0.1 mm. It looks like a humid evening! "
        "Would you like more details?"
    )

    # Find all match positions
    matches = list(SENTENCE_BOUNDARY_REGEX.finditer(text))
    # Should only split after '0.1 mm.' and after 'evening!'
    assert len(matches) == 2

    # Extract sentences using the boundary regex
    splits = []
    prev_end = 0
    for m in matches:
        splits.append(text[prev_end : m.start() + 1].strip())
        prev_end = m.end()
    splits.append(text[prev_end:].strip())

    assert len(splits) == 3
    assert splits[0] == "The weather in Bengaluru is 21.2°C (feels like 25.0°C), humidity 97%, wind 2.6 km/h, precipitation 0.1 mm."
    assert splits[1] == "It looks like a humid evening!"
    assert splits[2] == "Would you like more details?"


def test_sentence_speaker_buffers_and_splits():
    """Verify SentenceSpeaker splits streamed tokens into complete sentences without splitting decimals."""
    speaker = SentenceSpeaker()

    tokens = [
        "Current", " weather", ":", " 23.5", "°C", ". ",
        "Mumbai", " is", " 28.1", "°C", ".\n",
        "Which", " one", " do", " you", " prefer", "?"
    ]

    for t in tokens:
        speaker.feed_token(t)

    speaker.flush()

    sentences = []
    while not speaker.queue.empty():
        sentences.append(speaker.queue.get())

    # Stop background thread
    speaker.queue.put(None)

    assert len(sentences) == 3
    assert sentences[0] == "Current weather: 23.5°C."
    assert sentences[1] == "Mumbai is 28.1°C."
    assert sentences[2] == "Which one do you prefer?"


@patch("tts.sd.play")
@patch("tts.sd.wait")
def test_speak_runs_without_raising(mock_wait, mock_play):
    """Verify speak() executes cleanly on a short string without crashing."""
    speak("Hello, this is a test.")
    assert mock_play.called
    assert mock_wait.called


@patch("tts.sd.stop")
def test_sentence_speaker_interrupt(mock_stop):
    """Verify interrupt() clears the queue and calls sounddevice.stop()."""
    speaker = SentenceSpeaker()
    speaker.queue.put("Sentence 1")
    speaker.queue.put("Sentence 2")

    speaker.interrupt()

    # Queue should be empty and stop called
    assert speaker.queue.empty()
    assert mock_stop.called

    speaker.queue.put(None)  # clean up worker
