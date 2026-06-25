from crew.state import GraphState

_WAREHOUSE: dict[str, dict] = {
    "6601-000148": {"in_stock": True,  "qty": 4,  "price": 18.99},
    "6602-001655": {"in_stock": True,  "qty": 5,  "price": 12.49},
    "DC31-00054D": {"in_stock": True,  "qty": 12, "price": 34.99},
    "DC31-00187A": {"in_stock": False, "qty": 0,  "price": 41.50},
    "DC32-00007A": {"in_stock": True,  "qty": 20, "price": 9.99},
    "DC62-30314K": {"in_stock": True,  "qty": 3,  "price": 27.00},
    "DC62-30314L": {"in_stock": False, "qty": 0,  "price": 29.00},
    "DC66-00470B": {"in_stock": True,  "qty": 7,  "price": 15.75},
    "DC97-00252J": {"in_stock": False, "qty": 0,  "price": 22.00},
    "DC97-15459H": {"in_stock": True,  "qty": 2,  "price": 55.00},
    "DC97-15596A": {"in_stock": False, "qty": 0,  "price": 189.99},
    "DC97-16350V": {"in_stock": True,  "qty": 9,  "price": 11.25},
    "DC97-18682D": {"in_stock": True,  "qty": 6,  "price": 19.00},
    "DC97-19289F": {"in_stock": True,  "qty": 1,  "price": 38.50},
    "DC97-19289H": {"in_stock": False, "qty": 0,  "price": 40.00},
    "DC97-19639A": {"in_stock": True,  "qty": 3,  "price": 145.00},
    "DC97-21786A": {"in_stock": False, "qty": 0,  "price": 33.00},
    "DD31-00016A": {"in_stock": True,  "qty": 8,  "price": 29.99},
    "DD31-00016B": {"in_stock": True,  "qty": 4,  "price": 44.00},
    "WP89503":     {"in_stock": False, "qty": 0,  "price": 8.75},
}

_OUT_OF_STOCK = {"in_stock": False, "qty": 0}


def stock(state: GraphState) -> dict:
    skus = [
        row["sku"]
        for row in state.get("parts_results", [])
        if row.get("sku")
    ]
    return {
        "stock_results": {
            sku: _WAREHOUSE.get(sku, _OUT_OF_STOCK)
            for sku in skus
        }
    }
