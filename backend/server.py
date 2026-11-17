from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import json
import os
import urllib.request
import urllib.error
import ast
import operator
import re
import sqlite3
import base64
import zipfile
import io
import xml.etree.ElementTree as ET

MODEL = "gemini-3.5-flash-lite"
API_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"

KEY_FILE = Path.home() / ".config" / "nexora" / "gemini_api_key"


SUPABASE_URL = "https://uljaleshmbmewwcerpxd.supabase.co"
SUPABASE_PUBLISHABLE_KEY = "sb_publishable_CQKj0FxvLxtQ4SeV-CaccQ_ee7libZ_"


def verify_supabase_token(token):
    if not token:
        return None

    request = urllib.request.Request(
        SUPABASE_URL + "/auth/v1/user",
        headers={
            "apikey": SUPABASE_PUBLISHABLE_KEY,
            "Authorization": "Bearer " + token
        }
    )

    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            user = json.loads(response.read().decode("utf-8"))

        user_id = user.get("id")
        return user_id if isinstance(user_id, str) else None

    except (urllib.error.HTTPError, urllib.error.URLError, ValueError):
        return None



DB_FILE = Path(__file__).resolve().parent / "nexora_memory.db"


def init_memory():
    with sqlite3.connect(DB_FILE) as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)


def supabase_memory_request(method, path, token, payload=None):
    body = None
    headers = {
        "apikey": SUPABASE_PUBLISHABLE_KEY,
        "Authorization": "Bearer " + token,
        "Content-Type": "application/json"
    }

    if payload is not None:
        body = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        SUPABASE_URL + "/rest/v1/nexora_memories" + path,
        data=body,
        headers=headers,
        method=method
    )

    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            raw = response.read()
            return json.loads(raw.decode("utf-8")) if raw else None
    except (urllib.error.HTTPError, urllib.error.URLError) as error:
        print("Supabase memory request failed:", type(error).__name__)
        raise RuntimeError("Не удалось обратиться к облачной памяти Supabase")


def add_memory(content, token, user_id):
    content = content.strip()
    if not content or len(content) > 1000:
        raise ValueError("Запись должна содержать от 1 до 1000 символов")

    supabase_memory_request(
        "POST",
        "",
        token,
        {"user_id": user_id, "content": content}
    )


def get_memories(token, user_id):
    rows = supabase_memory_request(
        "GET",
        "?select=id,content&user_id=eq." + user_id + "&order=created_at.desc&limit=50",
        token
    )
    return [(row["id"], row["content"]) for row in reversed(rows or [])]


def delete_memory(memory_number, token, user_id):
    rows = get_memories(token, user_id)

    if memory_number < 1 or memory_number > len(rows):
        return False

    memory_id = rows[memory_number - 1][0]

    supabase_memory_request(
        "DELETE",
        "?id=eq." + memory_id + "&user_id=eq." + user_id,
        token
    )
    return True


def delete_all_memories(token, user_id):
    rows = get_memories(token, user_id)
    if rows:
        supabase_memory_request(
            "DELETE",
            "?user_id=eq." + user_id,
            token
        )
    return len(rows)


SYSTEM_PROMPT = """
Ты — NEXORA, внимательный русскоязычный ИИ-помощник.

Правила:
1. Внимательно анализируй вопрос перед ответом.
2. Не соглашайся автоматически, если пользователь ошибается.
3. Проверяй вычисления и логику.
4. Не выдумывай факты. Если не знаешь — скажи честно.
5. Разделяй факты и предположения.
6. Отвечай понятно и последовательно на русском языке.
7. Если вопрос неоднозначный, попроси уточнение.
8. Объясняй достаточно подробно, но без лишней воды.
"""

OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow
}

def calculate(node):
    if isinstance(node, ast.Expression):
        return calculate(node.body)

    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = calculate(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value

    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        left = calculate(node.left)
        right = calculate(node.right)

        if isinstance(node.op, ast.Pow) and abs(right) > 10:
            raise ValueError("Слишком большая степень")

        result = OPS[type(node.op)](left, right)

        if abs(result) > 10**100:
            raise ValueError("Слишком большое число")

        return result

    raise ValueError("Недопустимое выражение")


def exact_math(message):
    expr = message.replace("×", "*").replace("÷", "/").replace(",", ".")

    if not re.fullmatch(r"[0-9+\-*/().% \t]+", expr):
        return None

    try:
        return calculate(ast.parse(expr, mode="eval"))
    except Exception:
        return None


def choose_thinking_level(message):
    text = message.lower().strip().rstrip("!.? ")

    greetings = {
        "привет", "здравствуй", "здравствуйте",
        "доброе утро", "добрый день", "добрый вечер",
        "хай", "hello", "hi"
    }

    if text in greetings:
        return "minimal"

    complex_words = (
        "подробно", "проанализируй", "сравни",
        "докажи", "разработай", "напиши код",
        "исправь код", "пошагово", "объясни почему",
        "реши задачу"
    )

    if len(text) > 700 or any(w in text for w in complex_words):
        return "medium"

    return "low"


def ask_gemini(message, history=None, previous_interaction_id=None, files=None, access_token=None, user_id=None):
    key = os.environ.get("GEMINI_API_KEY", "").strip()

    if not key and KEY_FILE.exists():
        key = KEY_FILE.read_text(encoding="utf-8").strip()

    if not key:
        raise RuntimeError("Не задан Gemini API Key")
    if not key:
        raise RuntimeError("Файл Gemini API Key пуст")

    history = history if isinstance(history, list) else []
    context = []

    for item in history[-12:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = str(item.get("content", ""))[:6000]

        if role not in ("user", "assistant") or not content:
            continue

        name = "Пользователь" if role == "user" else "NEXORA"
        context.append(f"{name}: {content}")

    input_text = message
    if context and not previous_interaction_id:
        input_text = (
            "Предыдущая история диалога. Используй её как контекст:\n"
            + "\n".join(context)
            + "\n\nТекущее сообщение пользователя:\n"
            + message
        )


    files = files if isinstance(files, list) else []
    gemini_input = input_text

    if len(files) > 5:
        raise ValueError("Можно отправить не более 5 файлов")

    if files:
        gemini_input = []
        total_size = 0

        for item in files:
            if not isinstance(item, dict):
                raise ValueError("Некорректное вложение")

            name = str(item.get("name", "Файл"))[:120]
            mime = str(item.get("mime_type", "")).lower()
            encoded = item.get("data", "")

            if not isinstance(encoded, str):
                raise ValueError("Некорректные данные файла")

            try:
                raw = base64.b64decode(encoded, validate=True)
            except Exception:
                raise ValueError("Не удалось прочитать файл: " + name)

            total_size += len(raw)
            if total_size > 12 * 1024 * 1024:
                raise ValueError("Общий размер файлов превышает 12 МБ")

            if mime in ("image/jpeg", "image/png", "image/webp"):
                gemini_input.append({
                    "type": "image",
                    "data": encoded,
                    "mime_type": mime
                })

            elif mime in ("video/mp4", "video/webm"):
                gemini_input.append({
                    "type": "video",
                    "data": encoded,
                    "mime_type": mime
                })

            elif mime in ("audio/mp3", "audio/mpeg", "audio/wav", "audio/aac", "audio/m4a"):
                gemini_input.append({
                    "type": "audio",
                    "data": encoded,
                    "mime_type": mime
                })

            elif mime == "application/pdf":
                gemini_input.append({
                    "type": "document",
                    "data": encoded,
                    "mime_type": mime
                })

            elif mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                try:
                    with zipfile.ZipFile(io.BytesIO(raw)) as docx:
                        document = docx.read("word/document.xml")
                    root = ET.fromstring(document)
                    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
                    paragraphs = []
                    for paragraph in root.findall(".//w:p", ns):
                        text = "".join(
                            node.text or ""
                            for node in paragraph.findall(".//w:t", ns)
                        )
                        if text:
                            paragraphs.append(text)
                    file_text = "\\n".join(paragraphs)
                except Exception:
                    raise ValueError("Не удалось прочитать DOCX: " + name)

                if not file_text.strip():
                    raise ValueError("В документе нет читаемого текста: " + name)

                if len(file_text) > 1000000:
                    raise ValueError("Текст документа слишком большой: " + name)

                gemini_input.append({
                    "type": "text",
                    "text": "\\n\\nСодержимое документа Word " + name + ":\\n" + file_text
                })

            elif mime in ("text/plain", "text/csv", "application/json"):
                if len(raw) > 1024 * 1024:
                    raise ValueError("Текстовый файл слишком большой: " + name)
                try:
                    file_text = raw.decode("utf-8")
                except UnicodeDecodeError:
                    raise ValueError("Файл должен быть в кодировке UTF-8: " + name)

                gemini_input.append({
                    "type": "text",
                    "text": "\n\nСодержимое файла " + name + ":\n" + file_text
                })

            else:
                raise ValueError("Пока не поддерживается формат: " + name)

    if files:
        gemini_input.append({"type": "text", "text": input_text})

    saved_memories = get_memories(access_token, user_id)
    memory_context = ""

    if saved_memories:
        memory_context = (
            "\n\nСохранённая долговременная память пользователя. "
            "Учитывай эти факты в ответах, когда они относятся к вопросу. "
            "Не выдумывай дополнительные факты:\n"
            + "\n".join(
                f"- {content}" for _, content in saved_memories
            )
        )

    payload = {
        "model": MODEL,
        "input": gemini_input,
        "system_instruction": SYSTEM_PROMPT + memory_context,
        "generation_config": {
            "thinking_level": choose_thinking_level(message)
        }
    }

    if previous_interaction_id:
        payload["previous_interaction_id"] = previous_interaction_id

    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": key
        },
        method="POST"
    )

    import time

    started_at = time.monotonic()

    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))
            elapsed = time.monotonic() - started_at
            print(f"Время ответа Gemini: {elapsed:.1f} сек.")
            break

        except urllib.error.HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            print("Gemini HTTP error:", error.code, details[:3000])

            if error.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise

            delay = 2 ** (attempt + 1)
            print(
                f"Gemini временно недоступна ({error.code}). "
                f"Повтор через {delay} сек. Попытка {attempt + 2}/3"
            )
            time.sleep(delay)

    if result.get("status") != "completed":
        raise RuntimeError(
            "Gemini не завершила ответ. Статус: "
            + str(result.get("status"))
        )

    parts = []
    for step in result.get("steps", []):
        if step.get("type") == "model_output":
            for part in step.get("content", []):
                if part.get("type") == "text":
                    parts.append(part.get("text", ""))

    answer = "\n".join(parts).strip()
    if not answer:
        raise RuntimeError("Gemini вернула пустой ответ")

    return answer, result.get('id')


class NexoraHandler(BaseHTTPRequestHandler):

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")

        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self):
        if self.path == "/":
            self.send_json({
                "name": "NEXORA",
                "status": "online",
                "model": MODEL
            })
        else:
            self.send_json({"error": "Not found"}, 404)

    def do_POST(self):
        if self.path != "/chat":
            self.send_json({"error": "Not found"}, 404)
            return

        try:
            auth_header = self.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                self.send_json({"error": "Требуется авторизация через Google"}, 401)
                return

            access_token = auth_header[7:].strip()
            user_id = verify_supabase_token(access_token)

            if not user_id:
                self.send_json({"error": "Сессия недействительна. Войдите снова."}, 401)
                return

            length = int(self.headers.get("Content-Length", 0))

            if length <= 0 or length > 18_000_000:
                self.send_json({"error": "Некорректный размер запроса"}, 400)
                return

            data = json.loads(self.rfile.read(length))
            message = str(data.get("message", "")).strip()

            if not message:
                self.send_json({"error": "Пустое сообщение"}, 400)
                return

            if message == "/memories":
                rows = get_memories(access_token, user_id)
                text = (
                    "\n".join(
                        f"{index}. {content}"
                        for index, (_, content) in enumerate(rows, 1)
                    )
                    if rows else "Память пока пуста."
                )
                self.send_json({
                    "assistant": "NEXORA",
                    "message": text,
                    "model": "NEXORA Memory"
                })
                return

            if message.startswith("/remember "):
                content = message[len("/remember "):].strip()
                try:
                    add_memory(content, access_token, user_id)
                    answer = "Сохранила в долговременную память."
                except ValueError as error:
                    answer = str(error)

                self.send_json({
                    "assistant": "NEXORA",
                    "message": answer,
                    "model": "NEXORA Memory"
                })
                return

            if message.startswith("/forget "):
                target = message[len("/forget "):].strip()

                if target.lower() == "all":
                    count = delete_all_memories(access_token, user_id)
                    answer = f"Удалено записей: {count}."
                else:
                    try:
                        memory_id = int(target)
                        deleted = delete_memory(memory_id, access_token, user_id)
                        answer = (
                            "Запись удалена."
                            if deleted else "Запись с таким номером не найдена."
                        )
                    except ValueError:
                        answer = "Укажи номер записи или напиши /forget all."

                self.send_json({
                    "assistant": "NEXORA",
                    "message": answer,
                    "model": "NEXORA Memory"
                })
                return

            if message == "/remember" or message == "/forget":
                self.send_json({
                    "assistant": "NEXORA",
                    "message": "Команды: /remember текст, /memories, /forget номер, /forget all",
                    "model": "NEXORA Memory"
                })
                return

            result = exact_math(message)

            if result is not None:
                self.send_json({
                    "assistant": "NEXORA",
                    "message": f"Ответ: {result}",
                    "model": "NEXORA Calculator"
                })
                return

            files = data.get("files", [])
            if not isinstance(files, list):
                self.send_json({"error": "Некорректный список файлов"}, 400)
                return

            history = data.get("history", [])
            if not isinstance(history, list):
                history = []

            previous_interaction_id = data.get("interaction_id")
            if not isinstance(previous_interaction_id, str):
                previous_interaction_id = None

            answer, new_interaction_id = ask_gemini(
                message, history, previous_interaction_id, files,
                access_token, user_id
            )

            self.send_json({
                "assistant": "NEXORA",
                "message": answer,
                "model": MODEL,
                "interaction_id": new_interaction_id
            })

        except urllib.error.HTTPError as error:
            try:
                details = json.loads(error.read().decode("utf-8"))
                message = details.get("error", {}).get("message", "Ошибка Gemini API")
            except Exception:
                message = "Ошибка Gemini API"

            print("Gemini HTTP error:", error.code, message)
            self.send_json({
                "error": f"Gemini API: {message}"
            }, 502)

        except Exception as error:
            print("NEXORA error:", str(error))
            self.send_json({"error": str(error)}, 500)


init_memory()

print(f"NEXORA server starting on port 8000...")
print(f"AI model: {MODEL}")

HTTPServer(("0.0.0.0", int(os.environ.get("PORT", "8000"))), NexoraHandler).serve_forever()
