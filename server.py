import logging
import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

# Configure module logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

USE_LOCAL = os.environ.get("USE_LOCAL_ROUTER", "false").lower() == "true"
if not USE_LOCAL and not os.environ.get("MISTRAL_API_KEY"):
    logger.warning("MISTRAL_API_KEY not found in environment. Falling back to local regex-based router.")
    USE_LOCAL = True

if USE_LOCAL:
    logger.info("Initializing with Local Regex-Based Router...")
    from local_steering import app as steering_app, response_model
else:
    logger.info("Initializing with LLM-Based Router (Mistral)...")
    from steering import app as steering_app, response_model

app = FastAPI(title="LLM Activation Steering API")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    """Data model for incoming chat requests."""
    message: str

@app.post("/chat")
def chat(request: ChatRequest):
    """
    Endpoint to process a chat message through the LangGraph routing agent
    and the dynamically steered language model.
    """
    try:
        inputs = {"messages": [HumanMessage(content=request.message)]}
        
        # Invoke the LangGraph execution graph
        result = steering_app.invoke(inputs)
        
        # The graph returns a dict with a "messages" list. The last message is the final AIMessage.
        last_message = result["messages"][-1]
        
        # Retrieve current steering state to drive the frontend visualization
        steering_state = dict(response_model.last_active_steering)
        
        return {
            "response": last_message.content,
            "active_steering": steering_state
        }
    except Exception as e:
        logger.exception(f"An error occurred while processing the chat request: {request.message}")
        raise HTTPException(status_code=500, detail="Internal server error during chat processing.")

# Ensure static directory exists before mounting
if not os.path.exists("static"):
    os.makedirs("static")

# Mount the static frontend directory
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    port = 8000
    
    # Optional: Start ngrok for easy external sharing (useful for demos)
    try:
        from pyngrok import ngrok
        # Open an HTTP tunnel on the default port 8000
        public_url = ngrok.connect(port).public_url
        logger.info(f"ngrok tunnel created: \"{public_url}\" -> \"http://127.0.0.1:{port}\"")
    except ImportError:
        logger.info("pyngrok not installed, skipping ngrok tunneling.")
    except Exception as e:
        logger.warning(f"ngrok connection failed: {e}")

    logger.info(f"Starting Uvicorn server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
