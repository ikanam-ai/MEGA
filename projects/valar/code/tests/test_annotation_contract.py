import pytest
from valar.annotators.valuellama import build_prompt, parse_output
from valar.value_space.schwartz import SCHWARTZ_10, circumplex_distance


class TestBuildPrompt:
    def test_returns_two_messages(self):
        msgs = build_prompt("Test text.")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_user_message_contains_text(self):
        msgs = build_prompt("Важный текст")
        assert "Важный текст" in msgs[1]["content"]

    def test_all_values_listed(self):
        msgs = build_prompt("x")
        for v in SCHWARTZ_10:
            assert v in msgs[1]["content"]


class TestParseOutput:
    def test_valid_json(self):
        scores_dict = {v: 0.5 for v in SCHWARTZ_10}
        import json
        out = json.dumps(scores_dict)
        result = parse_output(out)
        assert result["parse_ok"] is True
        assert result["scores"] is not None
        assert len(result["scores"]) == 10

    def test_clamps_out_of_range(self):
        import json
        scores_dict = {v: 1.5 for v in SCHWARTZ_10}
        result = parse_output(json.dumps(scores_dict))
        assert all(v <= 1.0 for v in result["scores"].values() if v is not None)

    def test_empty_string_fails(self):
        result = parse_output("")
        assert result["parse_ok"] is False
        assert result["scores"] is None

    def test_truncated_json_handled(self):
        result = parse_output('{"Self-Direction": 0.8, "Stimulation": 0.3')
        assert result["parse_ok"] is False


class TestCircumplexDistance:
    def test_adjacent_values_distance_1(self):
        assert circumplex_distance("Self-Direction", "Stimulation") == 1

    def test_same_value_distance_0(self):
        assert circumplex_distance("Hedonism", "Hedonism") == 0

    def test_opposite_values_max_distance(self):
        assert circumplex_distance("Self-Direction", "Security") == 5
