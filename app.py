import os
import json
import threading
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
import anthropic

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=os.path.join(BASE_DIR, "static"))

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
  "multiSelect": true or false,
  "inputType": "text" | "choice" | "none",
  "finalPrompt": "The complete, ready-to-use prompt (only include when stage is final)",
  "progress": 0-100
}

Rules:
- Ask ONE question at a time max
- Keep messages warm, clear, and encouraging
- Options array should have 3-5 helpful suggestions when relevant, otherwise empty
- Set multiSelect to true when the question benefits from multiple answers (e.g. tone, audience, goals, features). Set to false for single-answer questions (e.g. format, length, yes/no)
- Options are always editable suggestions — the user may modify them before sending
- Progress should reflect how close we are to the final prompt (0 at start, 100 when final)
- When you have enough info, move to "final" and produce a masterfully crafted prompt IMMEDIATELY — never announce that you are about to produce the prompt, never ask for confirmation, never say "let me put it together" or similar. Just set stage to "final" and include the finalPrompt in the same response.
- The finalPrompt should be detailed, structured, and immediately usable
- Never produce JSON with syntax errors"""


CHECKPOINT_PROMPT = """You are PromptForge, a prompt engineering assistant. Based on the conversation so far, generate a draft prompt that captures everything learned up to this point.

This is a CHECKPOINT — not the final prompt. It should:
- Reflect all context and answers gathered so far
- Be well-structured and already usable
- Be clearly labeled as a work-in-progress that can be improved further

Respond ONLY with a JSON object:
{
  "checkpointPrompt": "the draft prompt text based on conversation so far",
  "summary": "one sentence describing what this draft captures"
}

No preamble, no explanation outside the JSON."""


OPTIMIZER_PROMPT = """You are an expert prompt analyst and rewriter. Your job is to silently improve a prompt by identifying and fixing any gaps before it reaches the user.

Analyze the given prompt for these gap categories:
1. Missing context — background info the AI would need to respond accurately
2. Missing role/audience clarity — unclear who the AI is, who it's speaking to, or what expertise to assume
3. Missing constraints — unstated rules, format, length, or boundaries
4. Missing success criteria — undefined tone, goals, or quality markers
5. Missing edge case handling — what to do if input is ambiguous or incomplete

Then rewrite the prompt with all gaps filled in. Your rewrite should:
- Preserve the original intent and structure completely
- Add only what is genuinely missing — do not bloat it
- Be immediately usable without further editing
- If the prompt is already comprehensive, return it as-is with only minor polish

Respond ONLY with a JSON object in this exact shape:
{
  "improved": "the full rewritten prompt text",
  "changes": ["short description of each change made"] or []
}

No preamble, no explanation outside the JSON."""


def build_checkpoint(messages: list) -> dict:
    """Generate a checkpoint draft prompt from the conversation so far."""
    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            system=CHECKPOINT_PROMPT,
            messages=messages,
        )
        text = response.content[0].text
        clean = text.replace("```json", "").replace("```", "").strip()
        start = clean.find("{")
        end = clean.rfind("}") + 1
        if start != -1 and end > start:
            clean = clean[start:end]
        return json.loads(clean)
    except Exception:
        return {}


def optimize_prompt(raw_prompt: str) -> tuple[str, list]:
    """Run the completeness optimizer on a draft prompt."""
    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4000,
            system=OPTIMIZER_PROMPT,
            messages=[
                {"role": "user", "content": f"Analyze and improve this prompt:\n\n{raw_prompt}"}
            ],
        )
        text = response.content[0].text
        clean = text.replace("```json", "").replace("```", "").strip()
        start = clean.find("{")
        end = clean.rfind("}") + 1
        if start != -1 and end > start:
            clean = clean[start:end]
        result = json.loads(clean)
        improved = result.get("improved", "").strip()
        changes = result.get("changes", [])
        if improved and len(improved) > len(raw_prompt) * 0.5:
            return improved, changes
        return raw_prompt, []
    except Exception:
        return raw_prompt, []


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


# Shared cache for background tasks
task_cache = {}
task_lock = threading.Lock()


def run_task_async(task_id: str, fn, *args):
    """Run any function in background and cache result."""
    result = fn(*args)
    with task_lock:
        task_cache[task_id] = result


@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        messages = data.get("messages", [])
        prompt_id = data.get("promptId", "")
        question_count = data.get("questionCount", 0)

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=messages,
        )

        text = response.content[0].text
        clean = text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean)

        # Every 2 questions — trigger a background checkpoint
        if (parsed.get("stage") != "final" and
                question_count > 0 and
                question_count % 2 == 0 and
                prompt_id):
            checkpoint_id = f"{prompt_id}_cp_{question_count}"
            t = threading.Thread(
                target=run_task_async,
                args=(checkpoint_id, build_checkpoint, messages),
                daemon=True,
            )
            t.start()
            parsed["checkpointId"] = checkpoint_id

        # Final prompt — run optimizer in background
        if parsed.get("stage") == "final" and parsed.get("finalPrompt") and prompt_id:
            parsed["optimizing"] = True
            optimizer_id = f"{prompt_id}_opt"
            t = threading.Thread(
                target=run_task_async,
                args=(optimizer_id, lambda p: {"improved": optimize_prompt(p)[0], "changes": optimize_prompt(p)[1]}, parsed["finalPrompt"]),
                daemon=True,
            )
            t.start()
            parsed["optimizerId"] = optimizer_id

        return jsonify(parsed)

    except json.JSONDecodeError:
        return jsonify({"error": "Failed to parse AI response"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/task-result", methods=["GET"])
def task_result():
    """Poll for any background task result."""
    task_id = request.args.get("taskId", "")
    with task_lock:
        result = task_cache.pop(task_id, None)
    if result:
        return jsonify({"ready": True, "result": result})
    return jsonify({"ready": False})


@app.route("/api/run", methods=["POST"])
def run_prompt():
    try:
        data = request.get_json()
        prompt = data.get("prompt", "")

        def generate():
            with client.messages.stream(
                model="claude-sonnet-4-6",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                for text in stream.text_stream:
                    yield f"data: {json.dumps({'text': text})}\n\n"
            yield "data: [DONE]\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
