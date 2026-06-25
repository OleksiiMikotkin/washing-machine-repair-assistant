import os

from core.llm_gate import chat_completion
from crew.state import GraphState

_SYSTEM = """\
You are a visual assistant for washing machine spare parts.
Analyze the provided image and identify:
1. Part number or SKU — read it from any label on the part, or identify it from your knowledge if you recognize the part. Always state it on its own line: "Part number: <value>"
2. Part name and category (e.g. drain pump, door seal, drive belt)
3. Error codes shown on the display
4. Machine model number if visible on a label

If you cannot determine the part number, say so explicitly.\
"""


def vision(state: GraphState) -> dict:
    model = os.getenv("VISION_MODEL", "openai/gpt-4o")
    image_path = state.get("image_path")
    if not image_path:
        return {"image_description": None}

    response = chat_completion(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": state["query"]},
                    {"type": "image_url", "image_url": {"url": image_path}},
                ],
            },
        ],
        temperature=0,
    )

    description = response.choices[0].message.content or ""
    return {"image_description": description.strip()}
