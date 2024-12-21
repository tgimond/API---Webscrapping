import uvicorn
import os

from src.app import get_application

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath("src/config/firebase-credentials.json")

app = get_application()

if __name__ == "__main__":
    uvicorn.run("main:app",host="127.0.0.1", debug=True, reload=True, port=8080)
