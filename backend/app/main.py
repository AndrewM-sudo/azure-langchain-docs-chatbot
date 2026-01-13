from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import FRONTEND_ORIGIN
from backend.app.routers.health import router as health_router
from backend.app.routers.chat import router as chat_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("App starting up...")
    # Startup: init DB connections, load models, warm caches, etc.
    # Example: app.state.db = await connect_db(settings.DATABASE_URL)
    yield
    # Shutdown: close connections, flush buffers, etc.
    # Example: await app.state.db.close()

app = FastAPI(
    title="LangChain Azure OpenAI Chatbot Backend",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers (versioning is optional but common)
app.include_router(health_router, prefix="/v1", tags=["health"])
app.include_router(chat_router, prefix="/v1", tags=["chat"])