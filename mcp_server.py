from mcp.server.fastmcp import FastMCP

mcp = FastMCP("catalogue-produits")

PRODUITS = [
    {"id": 1, "nom": "Produit A", "prix": 10.99},
    {"id": 2, "nom": "Produit B", "prix": 19.99},
    {"id": 3, "nom": "Produit C", "prix": 5.49},
]

@mcp.tool()
def addition(a: int, b: int) -> int:
    """Additionne deux nombres."""
    return a + b

@mcp.tool()
def obtenir_produit(id: int) -> dict:
    """Retourne un produit à partir de son ID, ou lève une erreur si introuvable."""
    for produit in PRODUITS:
        if produit["id"] == id:
            return produit
    raise ValueError(f"Produit avec l'ID {id} introuvable.")

@mcp.resource("catalogue://tout-les-produits")
def tout_les_produits() -> str:
    """Retourne la liste de tous les produits."""
    return (PRODUITS)

@mcp.prompt()
def comparer_produits(id1: int, id2: int) -> str:
    """Construit un prompt pour comparer deux produits du catalogue."""
    return f"Compare le produit avec l'ID {id1} et le produit avec l'ID {id2}."

if __name__ == "__main__":
    mcp.run()