import re
from datetime import datetime

from flask import Flask, request, render_template_string
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage

from mcp_google_calendar import CalendarClient

app = Flask(__name__)

HTML_PAGE = """
<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <title>Agenda Chatbot</title>
  <style>
    body { font-family: Arial, sans-serif; padding: 24px; max-width: 900px; margin: auto; background: #f8f9fb; }
    h1 { color: #222; }
    .card { background: white; border-radius: 12px; padding: 20px; box-shadow: 0 4px 24px rgba(0,0,0,0.08); margin-bottom: 16px; }
    label { display: block; font-weight: 600; margin-bottom: 6px; }
    input, textarea, button { width: 100%; font-size: 16px; margin-bottom: 12px; }
    button { padding: 12px 20px; background: #2563eb; color: white; border: none; border-radius: 8px; cursor: pointer; }
    button:hover { background: #1d4ed8; }
    .message { padding: 14px; border-radius: 10px; margin-bottom: 16px; }
    .success { background: #ecfdf5; color: #166534; }
    .error { background: #fef2f2; color: #b91c1c; }
    ul { padding-left: 18px; }
    .event { margin-bottom: 10px; }
  </style>
</head>
<body>
  <h1>Agenda Chatbot</h1>
  <div class="card">
    <form method="post" action="/add">
      <label for="user_text">Entrez votre rendez-vous en langage naturel :</label>
      <textarea id="user_text" name="user_text" rows="3" placeholder="Par exemple : Réunion avec Hugo le 10/07/2026 à 14h30 pour finaliser le projet"></textarea>
      <button type="submit">Ajouter au calendrier</button>
    </form>
    {% if message %}
      <div class="message {{ message_type }}">{{ message }}</div>
    {% endif %}
  </div>
  <div class="card">
    <h2>Prochains rendez-vous</h2>
    {% if events %}
      <ul>
        {% for event in events %}
          <li class="event"><strong>{{ event.summary }}</strong><br>
            {{ event.start }} - {{ event.end }}{% if event.location %} · {{ event.location }}{% endif %}
          </li>
        {% endfor %}
      </ul>
    {% else %}
      <p>Aucun événement futur trouvé.</p>
    {% endif %}
  </div>
</body>
</html>
"""


class AgendaWeb:
    def __init__(self, llm_model: str = "qwen:latest"):
        self.llm = ChatOllama(model=llm_model, temperature=0)
        self.history = []
        self.calendar = CalendarClient()
        self.authenticated = False

    def ask(self, question: str) -> str:
        self.history.append(HumanMessage(content=question))
        response = self.llm.invoke(self.history)
        self.history.append(AIMessage(content=response.content))
        return response.content.strip()

    def interpret_user_request(self, user_text: str) -> dict:
        prompt = (
            "Tu es un assistant qui convertit une phrase en données de rendez-vous. "
            "Si la phrase ne contient pas assez d'informations, réponds uniquement 'manquant'.\n"
            f"Texte: {user_text}\n"
            "Réponds sous la forme suivante :\n"
            "summary: <titre>\nstart: <date ISO> <heure>\nend: <date ISO> <heure>\nlocation: <lieu>\ndescription: <texte>\n"
        )
        response = self.ask(prompt)
        result = {
            "summary": "",
            "start": "",
            "end": "",
            "location": "",
            "description": "",
            "missing": False,
        }
        for line in response.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().lower()
            value = value.strip()
            if key in result:
                result[key] = value
        if not result["summary"] or not result["start"] or not result["end"]:
            result["missing"] = True
        return result

    def build_event_body(self, parsed: dict) -> dict:
        start_dt = datetime.fromisoformat(parsed["start"])
        end_dt = datetime.fromisoformat(parsed["end"])
        return self.calendar.build_event(
            summary=parsed["summary"],
            start_iso=start_dt.isoformat(),
            end_iso=end_dt.isoformat(),
            description=parsed["description"],
            location=parsed["location"],
        )

    def ensure_authenticated(self) -> None:
        if not self.authenticated:
            self.calendar.authenticate()
            self.authenticated = True

    def list_events(self, max_results: int = 10) -> list[dict]:
        self.ensure_authenticated()
        raw_events = self.calendar.list_events(max_results)
        events = []
        for event in raw_events:
            start = event["start"].get("dateTime", event["start"].get("date", ""))
            end = event["end"].get("dateTime", event["end"].get("date", ""))
            events.append(
                {
                    "summary": event.get("summary", "Sans titre"),
                    "start": start,
                    "end": end,
                    "location": event.get("location", ""),
                }
            )
        return events

    def add_event(self, user_text: str) -> dict:
        self.ensure_authenticated()
        parsed = self.interpret_user_request(user_text)
        if parsed["missing"]:
            raise ValueError(
                "Impossible de reconnaître toutes les informations du rendez-vous. "
                "Précise le titre, la date et l'heure."
            )
        event_body = self.build_event_body(parsed)
        return self.calendar.create_event(event_body)


agenda = AgendaWeb()


@app.route("/", methods=["GET"])
def index():
    try:
        events = agenda.list_events(10)
        return render_template_string(HTML_PAGE, message=None, message_type="", events=events)
    except FileNotFoundError as exc:
        return render_template_string(
            HTML_PAGE,
            message="Fichier credentials.json introuvable. Placez-le dans le dossier du projet.",
            message_type="error",
            events=[],
        )


@app.route("/add", methods=["POST"])
def add_event():
    user_text = request.form.get("user_text", "").strip()
    if not user_text:
        return render_template_string(
            HTML_PAGE,
            message="Veuillez saisir une phrase décrivant le rendez-vous.",
            message_type="error",
            events=agenda.list_events(10) if agenda.authenticated else [],
        )

    try:
        created = agenda.add_event(user_text)
        events = agenda.list_events(10)
        return render_template_string(
            HTML_PAGE,
            message=f"Rendez-vous créé : {created.get('htmlLink', 'événement ajouté')}",
            message_type="success",
            events=events,
        )
    except FileNotFoundError:
        return render_template_string(
            HTML_PAGE,
            message="Fichier credentials.json introuvable. Placez-le dans le dossier du projet.",
            message_type="error",
            events=[],
        )
    except ValueError as exc:
        events = agenda.list_events(10) if agenda.authenticated else []
        return render_template_string(
            HTML_PAGE,
            message=str(exc),
            message_type="error",
            events=events,
        )
    except Exception as exc:
        events = agenda.list_events(10) if agenda.authenticated else []
        return render_template_string(
            HTML_PAGE,
            message=f"Erreur lors de la création du rendez-vous : {exc}",
            message_type="error",
            events=events,
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
