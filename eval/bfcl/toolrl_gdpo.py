import hashlib
import json
import os
import re

from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from bfcl_eval.model_handler.utils import (
    convert_to_function_call,
    func_doc_language_specific_pre_processing,
)
from overrides import override


TOOLRL_SYSTEM_PREFIX = """You are a helpful multi-turn dialogue assistant capable of leveraging tool calls to solve user tasks and provide structured chat responses.

**Available Tools**
In your response, you can use the following tools:
{tools}

**Steps for Each Turn**
1. **Think:** Recall relevant context and analyze the current user goal.
2. **Decide on Tool Usage:** If a tool is needed, specify the tool and its parameters.
3. **Respond Appropriately:** If a response is needed, generate one while maintaining consistency across user queries.

**Output Format**
```plaintext
<think> Your thoughts and reasoning </think>
<tool_call>
{{"name": "Tool name", "parameters": {{"Parameter name": "Parameter content", "... ...": "... ..."}}}}
{{"name": "... ...", "parameters": {{"... ...": "... ...", "... ...": "... ..."}}}}
...
</tool_call>
<response> AI's final response </response>
```

**Important Notes**
1. You must always include the `<think>` field to outline your reasoning. Provide at least one of `<tool_call>` or `<response>`. Decide whether to use `<tool_call>` (possibly multiple times), `<response>`, or both.
2. You can invoke multiple tool calls simultaneously in the `<tool_call>` fields. Each tool call should be a JSON object with a "name" field and an "parameters" field containing a dictionary of parameters. If no parameters are needed, leave the "parameters" field an empty dictionary.
3. Refer to the previous dialogue records in the history, including the user's queries, previous `<tool_call>`, `<response>`, and any tool feedback noted as `<obs>` (if exists)."""

_TOOL_BLOCK_RE = re.compile(r"<tool_call>\s*\n(.*?)\n\s*</tool_call>", re.DOTALL)
_STRICT_RESPONSE_RE = re.compile(
    r"^<think>.*?</think>\n(?:"
    r"<tool_call>\n.*?\n</tool_call>(?:\n<response>.*?</response>)?"
    r"|<response>.*?</response>)$",
    re.DOTALL,
)


def _json_stream(text):
    """Decode one or more whitespace-separated JSON objects."""
    decoder = json.JSONDecoder()
    position = 0
    objects = []
    while position < len(text):
        while position < len(text) and text[position].isspace():
            position += 1
        if position == len(text):
            break
        value, position = decoder.raw_decode(text, position)
        objects.append(value)
    return objects


def extract_toolrl_calls(response, allow_arguments=True):
    calls = []
    try:
        for block in _TOOL_BLOCK_RE.findall(response):
            for call in _json_stream(block):
                if not isinstance(call, dict):
                    return []
                name = call.get("name")
                parameters = call.get("parameters")
                if parameters is None and allow_arguments:
                    parameters = call.get("arguments")
                if not isinstance(name, str) or not isinstance(parameters, dict):
                    return []
                calls.append({"name": name, "parameters": parameters})
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return calls


def is_valid_toolrl_response(response):
    if not isinstance(response, str) or _STRICT_RESPONSE_RE.fullmatch(response) is None:
        return False
    if response.count("<think>") != 1 or response.count("</think>") != 1:
        return False
    if response.count("<response>") != response.count("</response>"):
        return False
    if response.count("<response>") > 1:
        return False
    if "<tool_call>" not in response:
        return True
    if response.count("<tool_call>") != 1 or response.count("</tool_call>") != 1:
        return False
    return bool(extract_toolrl_calls(response, allow_arguments=False))


def paired_question_seed(base_seed, question_id):
    """Map a frozen evaluation seed and BFCL id to a stable vLLM seed."""
    payload = f"bfcl-v3-toolrl:{int(base_seed)}:{question_id}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**31 - 1)


def _tool_parameters(function):
    parameter_schema = function.get("parameters", {})
    properties = parameter_schema.get("properties", {})
    converted = {}
    for name, schema in properties.items():
        ordered = {}
        if "description" in schema:
            ordered["description"] = schema["description"]
        if "type" in schema:
            ordered["type"] = schema["type"]
        ordered["default"] = schema.get("default", "")
        for key, value in schema.items():
            if key not in ordered:
                ordered[key] = value
        converted[name] = ordered
    return converted


def _format_tools(functions):
    rendered = []
    for index, function in enumerate(functions, start=1):
        rendered.extend(
            [
                f"{index}. Name: {function['name']}",
                f"Description: {function.get('description', '')}",
                "Parameters: "
                + json.dumps(
                    _tool_parameters(function), ensure_ascii=False, separators=(", ", ": ")
                ),
            ]
        )
    return "\n".join(rendered)


def _assistant_history(content):
    if any(tag in content for tag in ("<think>", "<tool_call>", "<response>")):
        return content
    return f"<response> {content} </response>"


def _tool_observations(messages, start):
    observations = []
    position = start
    while position < len(messages) and messages[position]["role"] == "tool":
        message = messages[position]
        name = message.get("name", "tool")
        if isinstance(name, dict) and len(name) == 1:
            name = next(iter(name))
        try:
            result = json.loads(message["content"])
        except (TypeError, ValueError, json.JSONDecodeError):
            result = message["content"]
        observations.append({"name": str(name), "results": result})
        position += 1
    return observations, position


class ToolRLGDPOHandler(OSSHandler):
    """BFCL-v3 adapter for the ToolRL prompt and output protocol in GDPO."""

    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)
        self.is_fc_model = True

    @override
    def decode_ast(self, result, language="Python"):
        return [
            {call["name"]: call["parameters"]}
            for call in extract_toolrl_calls(result)
        ]

    @override
    def decode_execute(self, result):
        return convert_to_function_call(self.decode_ast(result))

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions = func_doc_language_specific_pre_processing(
            test_entry["function"], test_entry["id"].rsplit("_", 1)[0]
        )
        inference_data = {"message": [], "function": functions}
        base_seed = os.getenv("BFCL_EVAL_SEED")
        if base_seed is not None:
            inference_data["sampling_seed"] = paired_question_seed(
                base_seed, test_entry["id"]
            )
            inference_data["sampling_request_index"] = 0
        top_p = os.getenv("BFCL_EVAL_TOP_P")
        if top_p is not None:
            inference_data["top_p"] = float(top_p)
        return inference_data

    @override
    def _format_prompt(self, messages, function):
        additional_system_messages = [
            message["content"] for message in messages if message["role"] == "system"
        ]
        system_prompt = TOOLRL_SYSTEM_PREFIX.format(tools=_format_tools(function))
        if additional_system_messages:
            system_prompt += "\n\n**Additional System Instructions**\n" + "\n".join(
                additional_system_messages
            )

        history = []
        position = 0
        while position < len(messages):
            message = messages[position]
            role = message["role"]
            if role == "system":
                position += 1
                continue
            if role == "user":
                history.append(f"<user> {message['content']} </user>")
                position += 1
                continue
            if role == "assistant":
                history.append(_assistant_history(message["content"]))
                position += 1
                continue
            if role == "tool":
                observations, position = _tool_observations(messages, position)
                history.append(
                    "<obs> "
                    + json.dumps(observations, ensure_ascii=False, separators=(", ", ": "))
                    + " </obs>"
                )
                continue
            raise ValueError(f"Unsupported BFCL message role: {role}")

        joined_history = "**Dialogue Records History**"
        if history:
            joined_history += "\n" + "\n\n".join(history)
        return (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{joined_history}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
