from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import src.calls.environment.configure

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

# register api calls
app.include_router(src.calls.environment.configure.router)


@app.get("/")
async def root():
    return {"message": "Hello Poker"}
