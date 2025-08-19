from fastapi import APIRouter
from app.api.v1.endpoints import rag, documents, vectors, health, admin

# Create main API router
api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(
    rag.router,
    prefix="/rag",
    tags=["RAG Operations"],
    responses={
        200: {"description": "Success"},
        400: {"description": "Bad Request"},
        401: {"description": "Unauthorized"},
        404: {"description": "Not Found"},
        500: {"description": "Internal Server Error"}
    }
)

api_router.include_router(
    documents.router,
    prefix="/documents",
    tags=["Document Management"],
    responses={
        200: {"description": "Success"},
        201: {"description": "Created"},
        400: {"description": "Bad Request"},
        401: {"description": "Unauthorized"},
        404: {"description": "Not Found"},
        500: {"description": "Internal Server Error"}
    }
)

api_router.include_router(
    vectors.router,
    prefix="/vectors",
    tags=["Vector Operations"],
    responses={
        200: {"description": "Success"},
        400: {"description": "Bad Request"},
        401: {"description": "Unauthorized"},
        404: {"description": "Not Found"},
        500: {"description": "Internal Server Error"}
    }
)

api_router.include_router(
    health.router,
    prefix="/health",
    tags=["Health & Monitoring"],
    responses={
        200: {"description": "Healthy"},
        503: {"description": "Service Unavailable"}
    }
)

api_router.include_router(
    admin.router,
    prefix="/admin",
    tags=["Administration"],
    responses={
        200: {"description": "Success"},
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        500: {"description": "Internal Server Error"}
    }
)
