def messages_to_openai(messages: list) -> list[dict]:
    role_map = {"human": "user", "ai": "assistant", "system": "system"}
    result = []
    for m in messages:
        role = role_map.get(getattr(m, "type", "human"), "user")
        content = m.content
        if isinstance(content, str):
            result.append({"role": role, "content": content})
    return result
