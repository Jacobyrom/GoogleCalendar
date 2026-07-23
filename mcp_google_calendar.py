import os
import pickle
from datetime import datetime
from typing import Any, Dict

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/calendar.events"]


class CalendarClient:
    def __init__(
        self,
        credentials_path: str = "credentials.json",
        token_path: str = "token.pickle",
        calendar_id: str = "primary",
        timezone: str = "Europe/Paris",
    ):
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.calendar_id = calendar_id
        self.timezone = timezone
        self.creds = None
        self.service = None

    def authenticate(self) -> None:
        if os.path.exists(self.token_path):
            with open(self.token_path, "rb") as token_file:
                self.creds = pickle.load(token_file)

        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                if not os.path.exists(self.credentials_path):
                    raise FileNotFoundError(
                        f"Le fichier {self.credentials_path} est introuvable. "
                        "Créez-le depuis Google Cloud Console puis placez-le dans le dossier du projet."
                    )
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, SCOPES
                )
                self.creds = flow.run_local_server(port=0)
            with open(self.token_path, "wb") as token_file:
                pickle.dump(self.creds, token_file)

        self.service = build("calendar", "v3", credentials=self.creds)

    def ensure_service(self) -> None:
        if self.service is None:
            self.authenticate()

    def list_events(self, max_results: int = 10):
        self.ensure_service()
        now = datetime.utcnow().isoformat() + "Z"
        events_result = (
            self.service.events()
            .list(
                calendarId=self.calendar_id,
                timeMin=now,
                maxResults=max_results,
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )
        return events_result.get("items", [])

    def create_event(self, event_body: Dict[str, Any]):
        self.ensure_service()
        return (
            self.service.events()
            .insert(calendarId=self.calendar_id, body=event_body)
            .execute()
        )

    def build_event(
        self,
        summary: str,
        start_iso: str,
        end_iso: str,
        description: str = "",
        location: str = "",
    ) -> Dict[str, Any]:
        event = {
            "summary": summary,
            "start": {"dateTime": start_iso, "timeZone": self.timezone},
            "end": {"dateTime": end_iso, "timeZone": self.timezone},
        }
        if description:
            event["description"] = description
        if location:
            event["location"] = location
        return event
