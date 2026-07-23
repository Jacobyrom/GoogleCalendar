from __future__ import annotations

import ast
import operator
import os
import re
from typing import List, TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

def get_llm():
    """Retourne le LLM utilisé par tous les agents (temperature=0 pour la stabilité)."""
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    return ChatOpenAI(model=model_name, temperature=0)


LLM = "qwen:latest"

def llm_call(prompt: str) -> str:
    global LLM
    if LLM is None:
        LLM = get_llm()
    return LLM.invoke(prompt).content.strip()


class State(TypedDict):
    question: str
    categorie: str
    expression: str
    response: str
    critique: str
    valid: bool
    attempts: int
    history: List[str]


MAX_ATTEMPTS = 2


ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

ALLOWED_UNARYOPS = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_eval(expr: str) -> float:
    """
    Évalue une expression arithmétique en toute sécurité.
    Autorisé : nombres, parenthèses, + - * / ** %
    Interdit : imports, appels de fonctions, noms de variables, attributs, etc.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Expression invalide : {e}")

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return node.value
            raise ValueError("Seuls les nombres sont autorisés.")
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in ALLOWED_BINOPS:
                raise ValueError(f"Opérateur non autorisé : {op_type.__name__}")
            return ALLOWED_BINOPS[op_type](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in ALLOWED_UNARYOPS:
                raise ValueError(f"Opérateur unaire non autorisé : {op_type.__name__}")
            return ALLOWED_UNARYOPS[op_type](_eval(node.operand))
        raise ValueError(f"Élément non autorisé dans l'expression : {type(node).__name__}")

    return _eval(tree)

ROUTER_PROMPT = """Tu es un classifieur strict. Classe la question de l'utilisateur dans UNE seule
des catégories suivantes, et réponds avec UN SEUL MOT, sans ponctuation ni explication :

- calcul   : la question demande un calcul mathématique explicite ou implicite (achats, sommes, etc.)
- recherche : la question demande une explication, une définition ou une connaissance générale
- ambigu   : la question est incomplète, vague, ou dépend d'un contexte manquant

Question : {question}

Réponds uniquement par : calcul, recherche ou ambigu."""


def router(state: State) -> State:
    raw = llm_call(ROUTER_PROMPT.format(question=state["question"]))
    categorie = raw.lower().strip()
    if "calcul" in categorie:
        categorie = "calcul"
    elif "recherche" in categorie:
        categorie = "recherche"
    else:
        categorie = "ambigu"

    history = state.get("history", [])
    history.append(f"router: catégorie détectée = {categorie}")

    return {**state, "categorie": categorie, "history": history, "attempts": 0}


def route_after_router(state: State) -> str:
    return state["categorie"]



EXTRACTION_PROMPT = """Transforme la question suivante en une expression mathématique Python simple.
Ne réponds PAS à la question, donne UNIQUEMENT l'expression, sans texte autour, sans "=", sans mot.
Utilise uniquement des chiffres et les opérateurs + - * / ** %.

Exemples :
Question : Calcule moi 10 fois 50
Expression : 10 * 50

Question : J'achète 3 produits à 19.99 euros
Expression : 3 * 19.99

Question : {question}
Expression :"""


def extracteur_calcul(state: State) -> State:
    expression = llm_call(EXTRACTION_PROMPT.format(question=state["question"]))
    expression = expression.strip().strip("`").strip()

    history = state["history"]
    history.append(f"extracteur_calcul: expression = {expression}")

    return {**state, "expression": expression, "history": history}


def agent_calcul(state: State) -> State:
    history = state["history"]
    try:
        resultat = safe_eval(state["expression"])
        response = f"Le résultat est {resultat}."
        history.append(f"agent_calcul: résultat = {resultat}")
    except Exception as e:
        response = f"Erreur de calcul : {e}"
        history.append(f"agent_calcul: échec ({e})")

    return {**state, "response": response, "history": history}



RECHERCHE_PROMPT = """Réponds à la question suivante de manière pédagogique et concise,
en respectant STRICTEMENT ce format (avec ces trois titres) :

Définition :
...

Exemple :
...

À retenir :
...

Question : {question}"""


def agent_recherche(state: State) -> State:
    response = llm_call(RECHERCHE_PROMPT.format(question=state["question"]))
    history = state["history"]
    history.append("agent_recherche: réponse générée")
    return {**state, "response": response, "history": history}



AMBIGU_PROMPT = """La question suivante est trop vague ou incomplète pour y répondre directement.
Rédige UNE question de clarification claire et polie à poser à l'utilisateur,
sans essayer de répondre à sa question initiale.

Question de l'utilisateur : {question}"""


def agent_ambigu(state: State) -> State:
    response = llm_call(AMBIGU_PROMPT.format(question=state["question"]))
    history = state["history"]
    history.append("agent_ambigu: question de clarification générée")
    return {**state, "response": response, "history": history}


def validateur(state: State) -> State:
    categorie = state["categorie"]
    response = state.get("response", "")
    history = state["history"]

    valid = False
    critique = ""

    if categorie == "calcul":
        if re.search(r"-?\d+(\.\d+)?", response) and "Erreur" not in response:
            valid = True
            critique = "Réponse valide : résultat numérique présent."
        else:
            critique = "Aucun résultat numérique valide trouvé dans la réponse."

    elif categorie == "recherche":
        texte = response.lower()
        has_def = "définition" in texte
        has_ex = "exemple" in texte
        has_retenir = "à retenir" in texte or "a retenir" in texte
        if has_def and has_ex and has_retenir:
            valid = True
            critique = "Réponse valide : définition, exemple et point à retenir présents."
        else:
            manquants = [
                nom for nom, present in
                [("définition", has_def), ("exemple", has_ex), ("à retenir", has_retenir)]
                if not present
            ]
            critique = f"Sections manquantes : {', '.join(manquants)}."

    elif categorie == "ambigu":
        if "?" in response:
            valid = True
            critique = "Réponse valide : il s'agit bien d'une question de clarification."
        else:
            critique = "La réponse ne contient pas de question de clarification."

    history.append(f"validateur: {'valide' if valid else 'invalide'} ({critique})")

    return {**state, "valid": valid, "critique": critique, "history": history}


def route_after_validation(state: State) -> str:
    if state["valid"]:
        return "ok"
    if state["attempts"] < MAX_ATTEMPTS:
        return "retry"
    return "fail"

CORRECTION_CALCUL_PROMPT = """La première extraction d'expression était incorrecte.
Critique : {critique}
Question originale : {question}
Expression précédente : {expression}

Donne une NOUVELLE expression mathématique Python correcte, sans texte autour."""

CORRECTION_RECHERCHE_PROMPT = """Ta réponse précédente était incomplète.
Critique : {critique}

Réécris une réponse complète en respectant STRICTEMENT ce format :

Définition :
...

Exemple :
...

À retenir :
...

Question : {question}
Réponse précédente : {response}"""

CORRECTION_AMBIGU_PROMPT = """Ta réponse précédente n'était pas une vraie question de clarification.
Critique : {critique}
Question originale : {question}

Rédige une véritable question de clarification (avec un point d'interrogation)."""


def correction(state: State) -> State:
    attempts = state["attempts"] + 1
    history = state["history"]
    history.append(f"correction: tentative {attempts}")
    categorie = state["categorie"]

    if categorie == "calcul":
        nouvelle_expr = llm_call(CORRECTION_CALCUL_PROMPT.format(
            critique=state["critique"],
            question=state["question"],
            expression=state["expression"],
        )).strip().strip("`")
        history.append(f"correction: nouvelle expression = {nouvelle_expr}")
        try:
            resultat = safe_eval(nouvelle_expr)
            response = f"Le résultat est {resultat}."
            history.append(f"correction: résultat = {resultat}")
        except Exception as e:
            response = f"Erreur de calcul : {e}"
            history.append(f"correction: échec ({e})")
        return {**state, "expression": nouvelle_expr, "response": response,
                "attempts": attempts, "history": history}

    elif categorie == "recherche":
        response = llm_call(CORRECTION_RECHERCHE_PROMPT.format(
            critique=state["critique"],
            question=state["question"],
            response=state["response"],
        ))
        history.append("correction: nouvelle réponse recherche générée")
        return {**state, "response": response, "attempts": attempts, "history": history}

    else:
        response = llm_call(CORRECTION_AMBIGU_PROMPT.format(
            critique=state["critique"],
            question=state["question"],
        ))
        history.append("correction: nouvelle question de clarification générée")
        return {**state, "response": response, "attempts": attempts, "history": history}


if __name__ == "__main__":
    exemples = [
        "Calcule 25 * 12 + 8",
        "Explique ce qu'est un reverse proxy",
        "Tu peux m'aider avec 10 serveurs ?",
        "Combien coûtent 3 produits à 19.99 euros ?",
    ]

    for q in exemples:
        print("=" * 70)
        print(f"Question : {q}")
        resultat = run(q)
        print(f"Catégorie : {resultat['categorie']}")
        print(f"Réponse   : {resultat['response']}")
        print("Historique :")
        for ligne in resultat["history"]:
            print(f"  - {ligne}")
        print()