from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend import Backend
import src.calls.environment.configure
import src.calls.environment.reset
import src.calls.environment.step
import src.calls.environment.delete

app = FastAPI()

origins = [
    "http://localhost:1234",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.backend = Backend()

# register api calls
app.include_router(src.calls.environment.configure.router)
app.include_router(src.calls.environment.reset.router)
app.include_router(src.calls.environment.step.router)
app.include_router(src.calls.environment.delete.router)




@app.get("/")
async def root():
    return {"message": "Hello Poker"}
