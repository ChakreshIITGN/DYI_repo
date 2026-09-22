"""
The hidden-message codec. No LLM anywhere in this file -- it is pure logic,
so it can be tested exactly.

THE SCHEME
    Inside the gap between two words:  ' ' = dash,  '!' = dot
    Each gap carries exactly ONE morse letter.
    Reading the gaps left to right spells the secret.

    Y = -.--  ->  " !  "
    E = .     ->  "!"
    S = ...   ->  "!!!"

    carrier "this is a secret" + secret "YES"
      -> "this !  is!a!!!secret"

CONSEQUENCE worth understanding: a plain single space is itself valid morse
(a lone dash = T). So there is no such thing as an "empty" gap. The carrier
must therefore have exactly len(secret) + 1 words -- every gap carries a
letter, or the decode picks up junk at the end.

    python morse.py            # runs the self-test
"""

DASH = " "
DOT = "!"

MORSE = {
    "A": ".-",    "B": "-...",  "C": "-.-.",  "D": "-..",   "E": ".",
    "F": "..-.",  "G": "--.",   "H": "....",  "I": "..",    "J": ".---",
    "K": "-.-",   "L": ".-..",  "M": "--",    "N": "-.",    "O": "---",
    "P": ".--.",  "Q": "--.-",  "R": ".-.",   "S": "...",   "T": "-",
    "U": "..-",   "V": "...-",  "W": ".--",   "X": "-..-",  "Y": "-.--",
    "Z": "--..",
    "0": "-----", "1": ".----", "2": "..---", "3": "...--", "4": "....-",
    "5": ".....", "6": "-....", "7": "--...", "8": "---..", "9": "----.",
}
FROM_MORSE = {v: k for k, v in MORSE.items()}


def letter_to_gap(letter):
    """'Y' -> ' !  '"""
    code = MORSE[letter.upper()]
    return "".join(DASH if c == "-" else DOT for c in code)


def gap_to_letter(gap):
    """' !  ' -> 'Y'. Returns None if the gap is not a valid letter."""
    code = "".join("-" if c == DASH else "." for c in gap)
    return FROM_MORSE.get(code)


def encode(carrier, secret):
    """
    carrier: a sentence whose words contain no spaces or '!'
    secret : the word to hide, one letter per gap

    len(words) must be len(secret) + 1.
    """
    words = carrier.split()
    secret = secret.upper()
    if len(words) != len(secret) + 1:
        raise ValueError(
            f"carrier needs exactly {len(secret) + 1} words for secret "
            f"{secret!r}, got {len(words)}: {words}"
        )
    out = words[0]
    for letter, word in zip(secret, words[1:]):
        out += letter_to_gap(letter) + word
    return out


def decode(text):
    """
    Walk the string, split it into word-runs and gap-runs, decode each gap.
    Returns (secret, gaps) so you can see what it saw.
    """
    gaps, current, in_gap = [], "", False
    for ch in text:
        if ch in (DASH, DOT):
            if not in_gap:
                in_gap, current = True, ""
            current += ch
        else:
            if in_gap:
                gaps.append(current)
                in_gap = False
    if in_gap:
        gaps.append(current)

    letters = [gap_to_letter(g) for g in gaps]
    return "".join(l or "?" for l in letters), gaps


RULES = f"""The message hides a word in the GAPS between its words.
Inside a gap, a space character means a morse DASH and an exclamation mark
means a morse DOT. Each gap contains exactly one morse letter.
Read the gaps from left to right to spell the hidden word."""


def _selftest():
    print("gap for each letter of YES:")
    for ch in "YES":
        print(f"  {ch} = {MORSE[ch]:<5} -> {letter_to_gap(ch)!r}")

    msg = encode("this is a secret", "YES")
    print(f"\nencoded: {msg!r}")
    got, gaps = decode(msg)
    print(f"gaps   : {[g for g in gaps]}")
    print(f"decoded: {got}")
    assert got == "YES", got

    for secret, carrier in [
        ("OK", "alpha beta gamma"),
        ("SOS", "one two three four"),
        ("HI", "red green blue"),
        ("A", "left right"),
    ]:
        enc = encode(carrier, secret)
        dec, _ = decode(enc)
        assert dec == secret, (secret, enc, dec)
        print(f"  {secret:<4} in {carrier!r:<28} -> {enc!r}")

    try:
        encode("too few words", "YES")
    except ValueError as e:
        print(f"\nlength guard works: {e}")

    print("\nnote: a lone space decodes to", repr(gap_to_letter(" ")),
          "-- which is why carrier length is enforced.")
    print("ALL TESTS PASS")


if __name__ == "__main__":
    _selftest()