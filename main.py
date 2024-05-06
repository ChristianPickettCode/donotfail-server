from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()


class Item(BaseModel):
    name: str
    price: float

# Create


@app.post("/items/")
async def create_item(item: Item):
    # Logic to create item
    return {"message": "Item created successfully"}

# Read


@app.get("/items/{item_id}")
async def read_item(item_id: int):
    # Logic to retrieve item with given item_id
    return {"item_id": item_id}

# Update


@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item):
    # Logic to update item with given item_id
    return {"message": "Item updated successfully"}

# Delete


@app.delete("/items/{item_id}")
async def delete_item(item_id: int):
    # Logic to delete item with given item_id
    return {"message": "Item deleted successfully"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
