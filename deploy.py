import uvicorn
from main import app

if __name__ == "__main__":
    # For deployment
    uvicorn.run(app, host="0.0.0.0", port=8000)