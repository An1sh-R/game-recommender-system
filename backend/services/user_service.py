from typing import Dict, Optional

import bcrypt
from psycopg2.errors import UniqueViolation

from backend.database import get_db_cursor


# Profile keys used by both save/fetch operations.
TRAIT_KEYS = (
    "exploration",
    "story",
    "challenge",
    "strategy",
    "social",
    "relaxation",
)


CREATE_USER_SQL = """
INSERT INTO users (username, password_hash)
VALUES (%s, %s)
RETURNING id;
"""

AUTHENTICATE_USER_SQL = """
SELECT id, password_hash
FROM users
WHERE username = %s;
"""

UPSERT_PROFILE_SQL = """
INSERT INTO user_profiles (
    user_id, exploration, story, challenge, strategy, social, relaxation
)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (user_id)
DO UPDATE SET
    exploration = EXCLUDED.exploration,
    story = EXCLUDED.story,
    challenge = EXCLUDED.challenge,
    strategy = EXCLUDED.strategy,
    social = EXCLUDED.social,
    relaxation = EXCLUDED.relaxation
RETURNING id;
"""

GET_PROFILE_SQL = """
SELECT user_id, exploration, story, challenge, strategy, social, relaxation
FROM user_profiles
WHERE user_id = %s;
"""


def _hash_password(password: str) -> str:
    """Hash a raw password using bcrypt."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify_password(password: str, password_hash: str) -> bool:
    """Verify a raw password against a bcrypt hash."""
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def create_user(username: str, password: str) -> Optional[int]:
    """
    Create a new user and return user_id.
    Returns None when username already exists.
    """
    password_hash = _hash_password(password)

    try:
        with get_db_cursor(commit=True) as cur:
            cur.execute(CREATE_USER_SQL, (username, password_hash))
            row = cur.fetchone()
            return int(row["id"]) if row else None
    except UniqueViolation:
        return None


def authenticate_user(username: str, password: str) -> Optional[int]:
    """
    Validate username/password and return user_id when valid.
    Returns None for invalid credentials.
    """
    with get_db_cursor() as cur:
        cur.execute(AUTHENTICATE_USER_SQL, (username,))
        row = cur.fetchone()

    if not row:
        return None

    if _verify_password(password, row["password_hash"]):
        return int(row["id"])
    return None


def save_user_profile(user_id: int, profile_dict: Dict[str, float]) -> int:
    """
    Create or update a user's trait profile.
    Missing traits default to 0.0.
    Returns the profile row id.
    """
    values = [float(profile_dict.get(key, 0.0)) for key in TRAIT_KEYS]

    with get_db_cursor(commit=True) as cur:
        cur.execute(UPSERT_PROFILE_SQL, (user_id, *values))
        row = cur.fetchone()
        if not row:
            raise Exception("Failed to save user profile.")
        else:
            return int(row["id"])


def get_user_profile(user_id: int) -> Optional[Dict[str, float]]:
    """
    Fetch and return a user's profile traits.
    Returns None if profile does not exist.
    """
    with get_db_cursor() as cur:
        cur.execute(GET_PROFILE_SQL, (user_id,))
        row = cur.fetchone()

    if not row:
        return None

    return {
        "user_id": int(row["user_id"]),
        "exploration": float(row["exploration"]),
        "story": float(row["story"]),
        "challenge": float(row["challenge"]),
        "strategy": float(row["strategy"]),
        "social": float(row["social"]),
        "relaxation": float(row["relaxation"]),
    }
