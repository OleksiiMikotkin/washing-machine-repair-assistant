import json

from core.llm_gate import chat_completion

_MAX_ITERATIONS = 5


def run_tool_loop(
    *,
    messages: list[dict],
    tools: list[dict],
    tool_registry: dict,
    model: str,
    max_iterations: int = _MAX_ITERATIONS,
) -> dict:
    """Generic tool-use loop for any agent.

    Returns {"answer": str, "tool_calls": list[dict]}
    where tool_calls is the full log of {tool, args, result} entries.
    """
    msgs = list(messages)
    tool_calls_log: list[dict] = []

    for iteration in range(1, max_iterations + 1):
        response = chat_completion(
            model=model,
            messages=msgs,
            tools=tools,
            tool_choice="required" if iteration == 1 else "auto",
            temperature=0.1,
        )
        msg = response.choices[0].message

        if msg.tool_calls:
            msgs.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            })
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                fn = tool_registry.get(tc.function.name)
                try:
                    result = fn(**args) if fn else {"error": f"unknown tool: {tc.function.name}"}
                except Exception as e:
                    result = {"error": str(e)}

                tool_calls_log.append({"tool": tc.function.name, "args": args, "result": result})
                msgs.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.function.name,
                    "content": json.dumps(result, ensure_ascii=False),
                })
            continue

        return {"answer": msg.content or "", "tool_calls": tool_calls_log}

    return {"answer": "[max iterations reached]", "tool_calls": tool_calls_log}
