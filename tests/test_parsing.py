from prompteval.parsing import normalise_urgency, parse_response


def test_basic_block():
    p = parse_response("reasoning...\nANSWER: C\nCONFIDENCE: 85")
    assert (p.answer, p.confidence, p.status) == ("C", 85.0, "ok")


def test_markdown_and_percent():
    p = parse_response("**ANSWER:** (b)\n**Confidence:** 72%")
    assert p.answer is None or p.answer == "B"  # lowercase letters are not accepted
    p = parse_response("**ANSWER:** (B)\n**CONFIDENCE:** 72%")
    assert (p.answer, p.confidence) == ("B", 72.0)


def test_last_occurrence_wins():
    p = parse_response("ANSWER: A\nCONFIDENCE: 50\nOn reflection:\nANSWER: D\nCONFIDENCE: 90")
    assert (p.answer, p.confidence) == ("D", 90.0)


def test_no_guessing_from_free_text():
    p = parse_response("I think the answer is B.")
    assert p.answer is None and p.status == "no_answer"


def test_word_starting_with_letter_is_not_an_answer():
    assert parse_response("ANSWER: Both are wrong").answer is None


def test_letter_with_option_text():
    assert parse_response("ANSWER: E) Corticosteroids\nCONFIDENCE: 60").answer == "E"


def test_confidence_out_of_range():
    p = parse_response("ANSWER: A\nCONFIDENCE: 150")
    assert p.status == "invalid_confidence" and p.confidence is None


def test_missing_confidence():
    assert parse_response("ANSWER: A").status == "no_confidence"


def test_truncation_flag():
    assert parse_response("ANSWER: A\nCONFIDENCE: 1", "max_tokens").truncated


def test_triage_and_references():
    txt = ("REFERENCES:\n- ESC 2023 guidelines\n- no verifiable source\n\n"
           "URGENCY: Non urgent\nNEXT_STEP: start oral antibiotics\nANSWER: B\nCONFIDENCE: 70")
    p = parse_response(txt)
    assert p.urgency == "non-urgent"
    assert p.next_step == "start oral antibiotics"
    assert p.references == ["ESC 2023 guidelines", "no verifiable source"]


def test_urgency_normalisation():
    assert normalise_urgency("emergency") == "emergency"
    assert normalise_urgency("Urgente") == "urgent"
    assert normalise_urgency("non-urgent") == "non-urgent"
    assert normalise_urgency("unclear") is None
