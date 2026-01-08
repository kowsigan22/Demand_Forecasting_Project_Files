from fastapi import FastAPI
from typing import Optional

# Initialize the FastAPI app
app = FastAPI()

# Define the GET route
@app.get("/process")
async def process_data(input1: str, input2: int):
    # Perform some operations (you can replace this with your logic)
    output1 = input1 * 2  # Example: multiply input1 by 2
    output2 = input2 + 5  # Example: add 5 to input2
    
    # Return the outputs as a dictionary (which FastAPI will return as JSON)
    return {"output1": output1, "output2": output2}