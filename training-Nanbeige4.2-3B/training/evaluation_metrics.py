"""Dependency-free, task-aware scoring for the public 1.0 offline dataset."""

import hashlib
import json
import re


def json_object(text):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def nested(value, path):
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def score_prediction(task, prediction, target):
    predicted, expected = json_object(prediction), json_object(target)
    fields = {"POLICY": "parameter_action.action", "Q3": "next_step.action",
              "Q6": "corrective_action.action", "Q7": "action", "Q8": "decision"}
    paths = {"action_match": fields.get(task), "next_tool_match": "next_tool" if task == "POLICY" else None}
    result = {"json_valid": predicted is not None, "exact_match": prediction.strip() == target.strip()}
    for metric, path in paths.items():
        expected_value = nested(expected, path) if path else None
        result[metric] = (nested(predicted, path) == expected_value) if expected_value is not None else None
    return result


def aggregate_scores(records):
    result = {}
    for key in ("json_valid", "exact_match", "action_match", "next_tool_match"):
        eligible = [row[key] for row in records if row[key] is not None]
        result[key + "_eligible"] = len(eligible)
        result[key + "_rate"] = sum(eligible) / len(eligible) if eligible else None
    return result


def select_rows(rows, samples_per_task, seed):
    if samples_per_task < 0:
        raise ValueError("samples_per_task must be nonnegative (0 means all)")
    selected = []
    for task in sorted({row["task"] for row in rows}):
        group = sorted((row for row in rows if row["task"] == task),
                       key=lambda row: hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest())
        selected.extend(group[:samples_per_task] if samples_per_task else group)
    if not selected:
        raise ValueError("Empty evaluation dataset")
    return selected
