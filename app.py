import os
import json
from flask import Flask, request, jsonify, send_from_directory
import anthropic

app = Flask(__name__, static_folder="static")

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are PromptForge, an expert AI prompt engineer and coach. Your job is to help users craft powerful, precise prompts through intelligent staged conversation.

When a user describes what they want to achieve, you will:
1. ANALYZE their goal and decide the best approach (wizard steps, clarifying questions, or iterative refinement)
2. GUIDE them stage by stage — never overwhelm with too many questions at once
3. BUILD toward a final, polished prompt

Your response must ALWAYS be valid JSON in this exact shape:
{
  "stage": "intake" | "clarify" | "refine" | "final",
  "message": "Your conversational message to the user",
  "question": "A single focused question to ask (omit if stage is final)",
  "options": ["optional", "suggested", "answers"] or [],
  "inputType": "text" | "choice" | "none",
  "finalPrompt": "The complete, ready-to-use prompt (only include when stage is final)",
  "progress": 0-100
}

Rules:
- Ask ONE question at a time max
- Keep messages warm, clear, and encouraging
- Options array should have 3-5 helpful suggestions when relevant, otherwise empty
- Progress should reflect how close we are to the final prompt (0 at start, 100 when final)
- When you have enough info, move to "final" and produce a masterfully crafted prompt
- The finalPrompt should be detailed, structured, and immediately usable
- Never produce JSON with syntax errors"""


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        messages = data.get("messages", [])

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=messages,
        )

        text = response.content[0].text
        # Strip markdown code fences if present
        clean = text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean)
        return jsonify(parsed)

    except json.JSONDecodeError:
        return jsonify({"error": "Failed to parse AI response"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
