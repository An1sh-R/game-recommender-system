from fastapi import APIRouter, HTTPException

from backend.schemas.auth import LoginRequest, RegisterRequest
from backend.services.user_service import authenticate_user, create_user


router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register")
def register(req: RegisterRequest):
    """
    Register a new user with username/password.
    Password is hashed via bcrypt in the service layer.
    """
    user_id = create_user(username=req.username, password=req.password)

    if user_id is None:
        raise HTTPException(status_code=409, detail="Username already exists")

    return {
        "success": True,
        "message": "Registration successful",
        "user_id": user_id,
    }


@router.post("/login")
def login(req: LoginRequest):
    """
    Validate credentials and return minimal login payload.
    """
    user_id = authenticate_user(username=req.username, password=req.password)

    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    return {
        "success": True,
        "message": "Login successful",
        "user_id": user_id,
    }
