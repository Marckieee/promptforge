import os
import json
import threading
import secrets
import psycopg
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context, session, redirect, url_for
from flask_dance.contrib.google import make_google_blueprint, google
from flask_dance.consumer import oauth_authorized
from werkzeug.middleware.proxy_fix import ProxyFix
import anthropic

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=os.path.join(BASE_DIR, "static"))
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = True

os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"

# Google OAuth blueprint
google_bp = make_google_blueprint(
    client_id=os.environ.get("GOOGLE_CLIENT_ID"),
    client_secret=os.environ.get("GOOGLE_CLIENT_SECRET"),
    scope=["openid", "https://www.googleapis.com/auth/userinfo.email", "https://www.googleapis.com/auth/userinfo.profile"],
)
app.register_blueprint(google_bp, url_prefix="/login")

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
DATABASE_URL = os.environ.get("DATABASE_URL")


# ── DATABASE ─────────────────────────────────────────────────
def get_db():
    return psycopg.connect(DATABASE_URL, row_factory=psycopg.rows.dict_row)

def init_db():
    try:
        conn = get_db()
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                session_id TEXT PRIMARY KEY,
                complexity  TEXT DEFAULT 'auto',
                target_ai   TEXT DEFAULT 'general',
                tone        TEXT DEFAULT 'balanced',
                verbosity   TEXT DEFAULT 'standard',
                updated_at  TIMESTAMP DEFAULT NOW()
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id          SERIAL PRIMARY KEY,
                google_id   TEXT UNIQUE NOT NULL,
                email       TEXT UNIQUE NOT NULL,
                name        TEXT,
                avatar      TEXT,
                created_at  TIMESTAMP DEFAULT NOW()
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS teams (
                id          SERIAL PRIMARY KEY,
                name        TEXT NOT NULL,
                owner_id    INTEGER REFERENCES users(id),
                created_at  TIMESTAMP DEFAULT NOW()
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS team_members (
                team_id    INTEGER REFERENCES teams(id) ON DELETE CASCADE,
                user_id    INTEGER REFERENCES users(id) ON DELETE CASCADE,
                role       TEXT DEFAULT 'member',
                joined_at  TIMESTAMP DEFAULT NOW(),
                PRIMARY KEY (team_id, user_id)
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS team_invites (
                id         SERIAL PRIMARY KEY,
                team_id    INTEGER REFERENCES teams(id) ON DELETE CASCADE,
                token      TEXT UNIQUE NOT NULL,
                email      TEXT,
                used       BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS team_prompts (
                id          SERIAL PRIMARY KEY,
                team_id     INTEGER REFERENCES teams(id) ON DELETE CASCADE,
                user_id     INTEGER REFERENCES users(id),
                goal        TEXT,
                prompt      TEXT NOT NULL,
                title       TEXT,
                created_at  TIMESTAMP DEFAULT NOW()
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS personal_prompts (
                id           SERIAL PRIMARY KEY,
                user_id      INTEGER REFERENCES users(id) ON DELETE CASCADE,
                goal         TEXT,
                prompt       TEXT NOT NULL,
                title        TEXT,
                session_data TEXT,
                created_at   TIMESTAMP DEFAULT NOW()
            )
        """)
        # Add session_data column if it doesn't exist (for existing tables)
        cur.execute("""
            ALTER TABLE personal_prompts
            ADD COLUMN IF NOT EXISTS session_data TEXT
        """)

        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"DB init error: {e}")

try:
    init_db()
except Exception:
    pass


# ── AUTH HELPERS ─────────────────────────────────────────────
def get_current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
        user = cur.fetchone()
        cur.close()
        conn.close()
        return dict(user) if user else None
    except Exception:
        return None

def upsert_user(google_id, email, name, avatar):
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO users (google_id, email, name, avatar)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (google_id) DO UPDATE SET
                email = EXCLUDED.email,
                name  = EXCLUDED.name,
                avatar= EXCLUDED.avatar
            RETURNING id
        """, (google_id, email, name, avatar))
        row = cur.fetchone()
        conn.commit()
        cur.close()
        conn.close()
        return row["id"] if row else None
    except Exception as e:
        print(f"Upsert user error: {e}")
        return None


# ── PREFS HELPERS ─────────────────────────────────────────────
def get_prefs(session_id: str) -> dict:
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT * FROM user_preferences WHERE session_id = %s", (session_id,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        return dict(row) if row else {}
    except Exception:
        return {}

def save_prefs(session_id: str, prefs: dict):
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO user_preferences (session_id, complexity, target_ai, tone, verbosity, updated_at)
            VALUES (%s, %s, %s, %s, %s, NOW())
            ON CONFLICT (session_id) DO UPDATE SET
                complexity = EXCLUDED.complexity,
                target_ai  = EXCLUDED.target_ai,
                tone       = EXCLUDED.tone,
                verbosity  = EXCLUDED.verbosity,
                updated_at = NOW()
        """, (
            session_id,
            prefs.get("complexity", "auto"),
            prefs.get("targetAI", "general"),
            prefs.get("tone", "balanced"),
            prefs.get("verbosity", "standard"),
        ))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Save prefs error: {e}")


# ── SYSTEM PROMPT ─────────────────────────────────────────────
def build_system_prompt(prefs: dict) -> str:
    complexity = prefs.get("complexity", "auto")
    target_ai  = prefs.get("target_ai", "general")
    tone       = prefs.get("tone", "balanced")
    verbosity  = prefs.get("verbosity", "standard")

    prefs_block = ""
    if complexity != "auto" or target_ai != "general" or tone != "balanced" or verbosity != "standard":
        prefs_block = f"""
## User preferences (apply these to every response)
- Complexity level: {complexity}
- Target AI: {target_ai}
- Tone: {tone}
- Verbosity: {verbosity}
"""

    return f"""You are PromptForge, an expert AI prompt engineer and coach. Your job is to help users craft powerful, precise prompts through intelligent staged conversation.

You are speaking to a mixed audience — complete beginners through experienced prompt engineers. Adapt your vocabulary, question depth, and final prompt complexity to match how detailed and technical the user's initial input is.
{prefs_block}
When a user describes what they want to achieve, you will:
1. ANALYZE their goal and decide the best approach
2. GUIDE them stage by stage — never overwhelm with too many questions at once
3. BUILD toward a final, polished prompt — do not simply restate the user's goal. The final prompt must be substantively richer.

## Few-shot examples of good question style
When a user says: "I want a prompt for writing emails"
Good question: "Who will be reading these emails — colleagues, clients, or cold prospects?"
Bad question: "What kind of emails do you want to write?"

Your response must ALWAYS be valid JSON:
{{
  "stage": "intake" | "clarify" | "refine" | "final",
  "message": "Your conversational message",
  "question": "A single focused question (omit if final)",
  "questionIndex": 1,
  "options": [] or ["opt1", "opt2"],
  "multiSelect": true or false,
  "inputType": "text" | "choice" | "none",
  "finalPrompt": "complete prompt (only when final)",
  "progress": 0-100
}}

Rules:
- Ask ONE question at a time
- Options must be specific and meaningfully different
- questionIndex increments by 1 each question
- When ready, produce finalPrompt IMMEDIATELY with no announcement
- finalPrompt must include role, context, constraints, and success criteria
- Never ask for info already provided
- Never produce invalid JSON"""


CHECKPOINT_PROMPT = """You are PromptForge. Generate a draft checkpoint prompt from the conversation so far.
Respond ONLY with JSON: {"checkpointPrompt": "...", "summary": "one sentence"}"""

OPTIMIZER_PROMPT = """You are an expert prompt analyst. Improve the given prompt by filling gaps in context, role clarity, constraints, success criteria, and edge cases.
Respond ONLY with JSON: {"improved": "full rewritten prompt", "changes": ["change 1", "change 2"]}"""



SPLITTER_PROMPT = """You are a prompt execution planner. Decide if a prompt needs splitting into parts.

ONLY split if the prompt explicitly requests ALL of these together:
- A multi-day schedule (4+ days) OR a very long structured guide with 4+ distinct sections
- AND would clearly produce more than 1000 words in a single response

DO NOT split:
- Simple questions or tasks
- Short prompts (under 100 words)
- Prompts asking for a single piece of content (email, post, analysis, code)
- Anything that can be answered well in one response

If splitting IS needed, create 2-3 parts maximum. Each part must be fully self-contained.

Respond ONLY with a JSON array. Examples:

No split needed: ["original prompt here"]

Split needed (6-day itinerary):
[
  "Overview and practical tips for the trip: [context from original prompt]",
  "Detailed itinerary for Days 1-3: [context from original prompt]",
  "Detailed itinerary for Days 4-6: [context from original prompt]"
]

IMPORTANT: Return the original prompt unchanged if no split is needed. Never split into more than 3 parts."""

# ── BACKGROUND HELPERS ────────────────────────────────────────
def build_checkpoint(messages):
    try:
        response = client.messages.create(model="claude-sonnet-4-6", max_tokens=2000, system=CHECKPOINT_PROMPT, messages=messages)
        text = response.content[0].text.replace("```json","").replace("```","").strip()
        start = text.find("{"); end = text.rfind("}") + 1
        return json.loads(text[start:end]) if start != -1 else {}
    except Exception:
        return {}

def optimize_prompt(raw_prompt):
    try:
        response = client.messages.create(model="claude-sonnet-4-6", max_tokens=4000, system=OPTIMIZER_PROMPT,
            messages=[{"role":"user","content":f"Improve this prompt:\n\n{raw_prompt}"}])
        text = response.content[0].text.replace("```json","").replace("```","").strip()
        start = text.find("{"); end = text.rfind("}") + 1
        result = json.loads(text[start:end]) if start != -1 else {}
        improved = result.get("improved","").strip()
        return (improved, result.get("changes",[])) if improved and len(improved) > len(raw_prompt)*0.5 else (raw_prompt,[])
    except Exception:
        return raw_prompt, []

task_cache = {}
task_lock  = threading.Lock()

def run_task_async(task_id, fn, *args):
    result = fn(*args)
    with task_lock:
        task_cache[task_id] = result


# ── AUTH ROUTES ───────────────────────────────────────────────
@app.route("/auth/google")
def auth_google():
    return redirect(url_for("google.login"))

@oauth_authorized.connect_via(google_bp)
def google_logged_in(blueprint, token):
    if not token:
        return False
    try:
        resp = blueprint.session.get("/oauth2/v2/userinfo")
        if not resp.ok:
            return False
        info = resp.json()
        user_id = upsert_user(
            google_id=info["id"],
            email=info["email"],
            name=info.get("name", ""),
            avatar=info.get("picture", ""),
        )
        session["user_id"] = user_id
        session.modified = True
    except Exception as e:
        print(f"Auth signal error: {e}")
    return False  # Don't save OAuth token to DB, we handle our own session

@app.route("/auth/logout")
def auth_logout():
    session.clear()
    return redirect("/")

@app.route("/api/me")
def api_me():
    user = get_current_user()
    if not user:
        return jsonify({"loggedIn": False})
    # Get user's teams
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            SELECT t.id, t.name, tm.role
            FROM teams t JOIN team_members tm ON t.id = tm.team_id
            WHERE tm.user_id = %s
        """, (user["id"],))
        teams = [dict(r) for r in cur.fetchall()]
        cur.close()
        conn.close()
    except Exception:
        teams = []
    return jsonify({"loggedIn": True, "user": dict(user), "teams": teams})


# ── TEAM ROUTES ───────────────────────────────────────────────
@app.route("/api/teams", methods=["POST"])
def create_team():
    user = get_current_user()
    if not user:
        return jsonify({"error": "Not logged in"}), 401
    data = request.get_json()
    name = data.get("name","").strip()
    if not name:
        return jsonify({"error": "Team name required"}), 400
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("INSERT INTO teams (name, owner_id) VALUES (%s, %s) RETURNING id", (name, user["id"]))
        team_id = cur.fetchone()["id"]
        cur.execute("INSERT INTO team_members (team_id, user_id, role) VALUES (%s, %s, 'owner')", (team_id, user["id"]))
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({"ok": True, "teamId": team_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/teams/<int:team_id>/invite", methods=["POST"])
def create_invite(team_id):
    user = get_current_user()
    if not user:
        return jsonify({"error": "Not logged in"}), 401
    data = request.get_json()
    email = data.get("email","").strip() or None
    token = secrets.token_urlsafe(16)
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("INSERT INTO team_invites (team_id, token, email) VALUES (%s, %s, %s)", (team_id, token, email))
        conn.commit()
        cur.close()
        conn.close()
        invite_url = f"{request.host_url}join/{token}"
        return jsonify({"ok": True, "inviteUrl": invite_url, "token": token})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/join/<token>")
def join_team(token):
    user = get_current_user()
    if not user:
        session["pending_invite"] = token
        return redirect("/auth/google")
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT * FROM team_invites WHERE token = %s AND used = FALSE", (token,))
        invite = cur.fetchone()
        if not invite:
            cur.close(); conn.close()
            return redirect("/?error=invalid_invite")
        cur.execute("INSERT INTO team_members (team_id, user_id) VALUES (%s, %s) ON CONFLICT DO NOTHING", (invite["team_id"], user["id"]))
        cur.execute("UPDATE team_invites SET used = TRUE WHERE token = %s", (token,))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Join team error: {e}")
    return redirect("/")

@app.route("/api/teams/<int:team_id>/prompts", methods=["GET"])
def get_team_prompts(team_id):
    user = get_current_user()
    if not user:
        return jsonify({"error": "Not logged in"}), 401
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            SELECT tp.*, u.name as saved_by
            FROM team_prompts tp LEFT JOIN users u ON tp.user_id = u.id
            WHERE tp.team_id = %s ORDER BY tp.created_at DESC
        """, (team_id,))
        prompts = [dict(r) for r in cur.fetchall()]
        cur.close()
        conn.close()
        return jsonify(prompts)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/teams/<int:team_id>/prompts", methods=["POST"])
def save_team_prompt(team_id):
    user = get_current_user()
    if not user:
        return jsonify({"error": "Not logged in"}), 401
    data = request.get_json()
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO team_prompts (team_id, user_id, goal, prompt, title)
            VALUES (%s, %s, %s, %s, %s) RETURNING id
        """, (team_id, user["id"], data.get("goal",""), data.get("prompt",""), data.get("title","")))
        prompt_id = cur.fetchone()["id"]
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({"ok": True, "id": prompt_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/teams/<int:team_id>/prompts/<int:prompt_id>", methods=["DELETE"])
def delete_team_prompt(team_id, prompt_id):
    user = get_current_user()
    if not user:
        return jsonify({"error": "Not logged in"}), 401
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("DELETE FROM team_prompts WHERE id = %s AND team_id = %s AND user_id = %s", (prompt_id, team_id, user["id"]))
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── PERSONAL LIBRARY ROUTES ──────────────────────────────────
@app.route("/api/library", methods=["GET"])
def get_library():
    user = get_current_user()
    if not user:
        return jsonify({"error": "Not logged in"}), 401
    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute("SELECT id, goal, prompt, title, session_data, created_at FROM personal_prompts WHERE user_id = %s ORDER BY created_at DESC", (user["id"],))
        prompts = [dict(r) for r in cur.fetchall()]
        cur.close(); conn.close()
        return jsonify(prompts)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/library", methods=["POST"])
def save_library():
    user = get_current_user()
    if not user:
        return jsonify({"error": "Not logged in"}), 401
    data = request.get_json()
    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute("""
            INSERT INTO personal_prompts (user_id, goal, prompt, title, session_data)
            VALUES (%s, %s, %s, %s, %s) RETURNING id
        """, (user["id"], data.get("goal",""), data.get("prompt",""), data.get("title",""), data.get("sessionData","")))
        pid = cur.fetchone()["id"]
        conn.commit(); cur.close(); conn.close()
        return jsonify({"ok": True, "id": pid})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/library/<int:prompt_id>", methods=["DELETE"])
def delete_library(prompt_id):
    user = get_current_user()
    if not user:
        return jsonify({"error": "Not logged in"}), 401
    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute("DELETE FROM personal_prompts WHERE id = %s AND user_id = %s", (prompt_id, user["id"]))
        conn.commit(); cur.close(); conn.close()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── MAIN ROUTES ───────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/api/preferences", methods=["GET"])
def get_preferences():
    session_id = request.args.get("sessionId", "default")
    return jsonify(get_prefs(session_id))

@app.route("/api/preferences", methods=["POST"])
def save_preferences():
    data = request.get_json()
    save_prefs(data.get("sessionId","default"), data)
    return jsonify({"ok": True})

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data           = request.get_json()
        messages       = data.get("messages", [])
        prompt_id      = data.get("promptId", "")
        question_count = data.get("questionCount", 0)
        session_id     = data.get("sessionId", "default")

        prefs  = get_prefs(session_id)
        system = build_system_prompt(prefs)

        response = client.messages.create(model="claude-sonnet-4-6", max_tokens=2000, system=system, messages=messages)
        text  = response.content[0].text.replace("```json","").replace("```","").strip()
        # Extract JSON robustly in case of extra text
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start != -1 and end > start:
            text = text[start:end]
        parsed = json.loads(text)

        if parsed.get("stage") != "final" and question_count > 0 and question_count % 2 == 1 and prompt_id:
            cp_id = f"{prompt_id}_cp_{question_count}"
            threading.Thread(target=run_task_async, args=(cp_id, build_checkpoint, messages), daemon=True).start()
            parsed["checkpointId"] = cp_id

        if parsed.get("stage") == "final" and parsed.get("finalPrompt") and prompt_id:
            parsed["optimizing"] = True
            opt_id = f"{prompt_id}_opt"
            threading.Thread(target=run_task_async,
                args=(opt_id, lambda p: {"improved": optimize_prompt(p)[0], "changes": optimize_prompt(p)[1]}, parsed["finalPrompt"]),
                daemon=True).start()
            parsed["optimizerId"] = opt_id

        return jsonify(parsed)

    except json.JSONDecodeError:
        return jsonify({"error": "Failed to parse AI response"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/task-result", methods=["GET"])
def task_result():
    task_id = request.args.get("taskId","")
    with task_lock:
        result = task_cache.pop(task_id, None)
    if result:
        return jsonify({"ready": True, "result": result})
    return jsonify({"ready": False})

def get_prompt_chunks(prompt: str) -> list[str]:
    """Split a complex prompt into sequential chunks if needed."""
    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            system=SPLITTER_PROMPT,
            messages=[{"role": "user", "content": f"Should this prompt be split? If yes, split it:\n\n{prompt}"}],
        )
        text = response.content[0].text.strip()

        # Strip markdown fences
        text = text.replace("```json", "").replace("```", "").strip()

        # Extract JSON array
        start = text.find("[")
        end   = text.rfind("]") + 1
        if start != -1 and end > start:
            chunks = json.loads(text[start:end])
            if isinstance(chunks, list) and len(chunks) > 1:
                print(f"Splitting into {len(chunks)} chunks")
                return chunks
            elif isinstance(chunks, list) and len(chunks) == 1:
                # Only one chunk returned — no split needed
                return [prompt]
    except Exception as e:
        print(f"Splitter error: {e}")
    return [prompt]


@app.route("/api/run", methods=["POST"])
def run_prompt():
    try:
        data   = request.get_json()
        prompt = data.get("prompt","")

        def generate():
            import time

            # Get chunks (or just the original if no split needed)
            chunks = get_prompt_chunks(prompt)
            total  = len(chunks)

            for i, chunk in enumerate(chunks):
                # Send section header if multiple chunks
                if total > 1:
                    section_num = f"Part {i+1} of {total}"
                    yield f"data: {json.dumps({'text': f'\n\n─── {section_num} ───\n\n'})}\n\n"

                last_heartbeat = time.time()

                with client.messages.stream(
                    model="claude-sonnet-4-6",
                    max_tokens=2000,
                    messages=[{"role": "user", "content": chunk}]
                ) as stream:
                    for text in stream.text_stream:
                        yield f"data: {json.dumps({'text': text})}\n\n"
                        now = time.time()
                        if now - last_heartbeat > 10:
                            yield 'data: {"heartbeat": true}\n\n'
                            last_heartbeat = now

            yield "data: [DONE]\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
                "Transfer-Encoding": "chunked",
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


# ── AI ROUTER ─────────────────────────────────────────────────
import urllib.request

GEMINI_API_KEY     = os.environ.get("GEMINI_API_KEY")
GROQ_API_KEY       = os.environ.get("GROQ_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
DEEPSEEK_API_KEY   = os.environ.get("DEEPSEEK_API_KEY")

# Model definitions
AI_MODELS = {
    "claude": {
        "name": "Claude Sonnet 4.6",
        "provider": "Anthropic",
        "icon": "🟠",
        "strengths": ["creative writing", "complex reasoning", "nuanced analysis", "coding", "general tasks"],
        "description": "Best for creative, complex reasoning and nuanced tasks"
    },
    "gemini": {
        "name": "Gemini 2.5 Flash",
        "provider": "Google",
        "icon": "🔵",
        "strengths": ["research", "factual questions", "travel", "science", "current events", "summarisation"],
        "description": "Best for research, factual queries and travel planning"
    },
    "llama": {
        "name": "Llama 3.3 70B",
        "provider": "OpenRouter",
        "icon": "🟣",
        "strengths": ["coding", "technical", "programming", "debugging", "fast responses"],
        "description": "Best for coding, technical tasks and fast responses"
    },
    "deepseek": {
        "name": "DeepSeek R1",
        "provider": "DeepSeek",
        "icon": "🟡",
        "strengths": ["math", "logic", "reasoning", "problem solving", "analysis"],
        "description": "Best for math, logical reasoning and structured problem solving"
    },
    "qwen": {
        "name": "Qwen3 Coder",
        "provider": "OpenRouter",
        "icon": "🟢",
        "strengths": ["writing", "business", "multilingual", "summarisation", "general"],
        "description": "Best for writing, business tasks and multilingual content"
    },
}

ROUTER_PROMPT = """You are an AI model router. Given a prompt, decide which AI model would produce the best result.

Available models:
- claude: Creative writing, complex reasoning, nuanced analysis, general tasks
- gemini: Research, factual questions, travel planning, science, summarisation
- llama: Coding, technical tasks, programming, debugging, fast responses
- deepseek: Math, logic, reasoning, structured problem solving, analysis
- qwen: Writing, business tasks, multilingual content, summarisation

Respond ONLY with a JSON object:
{
  "model": "claude" | "gemini" | "llama" | "deepseek" | "qwen",
  "reason": "One sentence explaining why this model fits the prompt best"
}"""


def route_prompt(prompt: str) -> dict:
    """Determine the best AI model for a given prompt using keyword matching (fast, no API call)."""
    prompt_lower = prompt.lower()

    # Keyword-based routing — instant, no API needed
    if any(w in prompt_lower for w in ["code", "python", "javascript", "debug", "function", "programming", "script", "sql", "error", "fix", "bug", "algorithm", "class", "variable"]):
        model_id = "llama"
    elif any(w in prompt_lower for w in ["math", "calculate", "equation", "proof", "theorem", "integral", "derivative", "statistics", "probability"]):
        model_id = "deepseek"
    elif any(w in prompt_lower for w in ["travel", "itinerary", "restaurant", "hotel", "flight", "research", "facts", "history", "science", "geography", "country", "city", "visit", "tour"]):
        model_id = "gemini"
    elif any(w in prompt_lower for w in ["write", "email", "business", "report", "summarise", "summarize", "essay", "blog", "content", "copy", "marketing", "proposal"]):
        model_id = "qwen"
    else:
        model_id = "claude"

    reasons = {
        "llama": "Prompt involves coding or technical tasks — Llama excels here",
        "deepseek": "Prompt requires logical reasoning or analysis — DeepSeek is optimised for this",
        "gemini": "Prompt involves research, travel or factual content — Gemini is best for this",
        "qwen": "Prompt involves writing or business content — Qwen handles this well",
        "claude": "General or creative task — Claude is the best fit",
    }

    return {
        "model": model_id,
        "reason": reasons[model_id],
        **AI_MODELS[model_id]
    }


def stream_gemini(prompt: str):
    """Stream response from Gemini 1.5 Flash."""
    import urllib.request, urllib.error
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse&key={GEMINI_API_KEY}"
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"maxOutputTokens": 4000}}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        buffer = ""
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            if line.startswith("data:"):
                data_str = line[5:].strip()
                if not data_str:
                    continue
                try:
                    data = json.loads(data_str)
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        for part in parts:
                            text = part.get("text", "")
                            if text:
                                yield text
                except Exception:
                    continue


def stream_groq(prompt: str):
    """Stream response from Llama via Groq."""
    import urllib.request
    url = "https://api.groq.com/openai/v1/chat/completions"
    body = json.dumps({
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": 4000,
    }).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}",
    }, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        for line in resp:
            line = line.decode("utf-8").strip()
            if line.startswith("data:") and line != "data: [DONE]":
                try:
                    data = json.loads(line[5:].strip())
                    delta = data["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta
                except Exception:
                    continue


def stream_openrouter(prompt: str, model: str):
    """Stream response from OpenRouter (DeepSeek or Qwen)."""
    import urllib.request
    model_map = {
        "deepseek": "deepseek/deepseek-r1:free",
        "qwen":     "qwen/qwen3-coder:free",
        "llama":    "meta-llama/llama-3.3-70b-instruct:free",
    }
    url = "https://openrouter.ai/api/v1/chat/completions"
    body = json.dumps({
        "model": model_map.get(model, "qwen/qwen-2.5-72b-instruct:free"),
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": 4000,
    }).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://promptforgebuild.up.railway.app",
        "X-Title": "PromptForge",
    }, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        for line in resp:
            line = line.decode("utf-8").strip()
            if not line or line == "data: [DONE]":
                continue
            if line.startswith("data:"):
                try:
                    data = json.loads(line[5:].strip())
                    choices = data.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    # Handle both content and reasoning_content (DeepSeek R1)
                    text = delta.get("content") or delta.get("reasoning_content") or ""
                    if text:
                        yield text
                except Exception:
                    continue


def stream_deepseek(prompt: str):
    """Stream response directly from DeepSeek API."""
    import urllib.request
    url = "https://api.deepseek.com/chat/completions"
    body = json.dumps({
        "model": "deepseek-reasoner",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": 8000,
    }).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    }, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        for line in resp:
            line = line.decode("utf-8").strip()
            if not line or line == "data: [DONE]":
                continue
            if line.startswith("data:"):
                try:
                    data = json.loads(line[5:].strip())
                    choices = data.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    # DeepSeek reasoner has reasoning_content and content
                    text = delta.get("content") or ""
                    if text:
                        yield text
                except Exception:
                    continue


@app.route("/api/route", methods=["POST"])
def route_and_run():
    """Route a prompt to the best AI and stream the response."""
    try:
        data   = request.get_json()
        prompt = data.get("prompt", "")
        force_model = data.get("model", None)  # allow user override

        # Route the prompt
        if force_model and force_model in AI_MODELS:
            routing = {"model": force_model, "reason": "User selected", **AI_MODELS[force_model]}
        else:
            routing = route_prompt(prompt)

        model = routing["model"]

        def generate():
            import time
            yield f"data: {json.dumps({'routing': routing})}\n\n"
            last_heartbeat = time.time()

            try:
                if model == "gemini" and GEMINI_API_KEY:
                    streamer = stream_gemini(prompt)
                elif model == "deepseek" and DEEPSEEK_API_KEY:
                    streamer = stream_deepseek(prompt)
                elif model in ("qwen", "llama") and OPENROUTER_API_KEY:
                    streamer = stream_openrouter(prompt, model)
                elif model == "claude":
                    with client.messages.stream(
                        model="claude-sonnet-4-6",
                        max_tokens=4000,
                        messages=[{"role": "user", "content": prompt}]
                    ) as stream:
                        for text in stream.text_stream:
                            yield f"data: {json.dumps({'text': text})}\n\n"
                            now = time.time()
                            if now - last_heartbeat > 10:
                                yield 'data: {"heartbeat": true}\n\n'
                                last_heartbeat = now
                    yield "data: [DONE]\n\n"
                    return
                else:
                    yield f"data: {json.dumps({'error': f'Model {model} not available'})}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                for text in streamer:
                    yield f"data: {json.dumps({'text': text})}\n\n"
                    now = time.time()
                    if now - last_heartbeat > 10:
                        yield 'data: {"heartbeat": true}\n\n'
                        last_heartbeat = now

            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

            yield "data: [DONE]\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"}
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500
