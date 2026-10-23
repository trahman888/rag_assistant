"""
Login page: sign up / log in before anyone sees the chat.

TWO MODES (picked automatically)
    * Supabase mode  — when SUPABASE_URL and SUPABASE_KEY are in Streamlit
      secrets (online) or environment variables (local). Accounts are real
      Supabase accounts (email + password). Use this for the deployed app.
    * Local mode     — when those aren't set. Accounts are saved in
      user_data/users.json on this computer, with passwords hashed. Handy for
      testing on your Mac without any setup. On Streamlit Cloud this file is
      wiped on restart, so don't use local mode online.

WHAT THE REST OF THE APP GETS
    st.session_state["user"] = {"id": "...", "email": "..."}
    The "id" never changes for a person, so it's used to name their folder,
    their storage prefix and their vector database collection.
"""
import hashlib
import json
import os
import re
import secrets
from pathlib import Path

import streamlit as st

USERS_FILE = Path(__file__).parent / "user_data" / "users.json"


# --------------------------------------------------------------------------- config
def _secret(name: str):
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name)


def supabase_enabled() -> bool:
    return bool(_secret("SUPABASE_URL") and _secret("SUPABASE_KEY"))


def supabase_client():
    """One Supabase client per browser session. It holds the logged-in user's
    token, so it must NOT be shared between users (no st.cache_resource here)."""
    if "sb" not in st.session_state:
        from supabase import create_client
        st.session_state["sb"] = create_client(_secret("SUPABASE_URL"), _secret("SUPABASE_KEY"))
    return st.session_state["sb"]


# --------------------------------------------------------------------------- local mode
def _hash(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 200_000).hex()


def _load_users() -> dict:
    return json.loads(USERS_FILE.read_text()) if USERS_FILE.exists() else {}


def _local_sign_up(email: str, password: str) -> dict:
    users = _load_users()
    if email in users:
        raise ValueError("An account with this email already exists.")
    salt = secrets.token_hex(16)
    users[email] = {"id": secrets.token_hex(8), "salt": salt, "hash": _hash(password, salt)}
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    USERS_FILE.write_text(json.dumps(users, indent=2))
    return {"id": users[email]["id"], "email": email}


def _local_log_in(email: str, password: str) -> dict:
    u = _load_users().get(email)
    if not u or not secrets.compare_digest(u["hash"], _hash(password, u["salt"])):
        raise ValueError("Wrong email or password.")
    return {"id": u["id"], "email": email}


# --------------------------------------------------------------------------- supabase mode
def _supabase_sign_up(email: str, password: str) -> dict | None:
    res = supabase_client().auth.sign_up({"email": email, "password": password})
    if res.session is None:   # project requires email confirmation
        return None
    return {"id": res.user.id, "email": res.user.email}


def _supabase_log_in(email: str, password: str) -> dict:
    res = supabase_client().auth.sign_in_with_password({"email": email, "password": password})
    return {"id": res.user.id, "email": res.user.email}


# --------------------------------------------------------------------------- public
def current_user() -> dict | None:
    return st.session_state.get("user")


def log_out() -> None:
    if supabase_enabled() and "sb" in st.session_state:
        try:
            supabase_client().auth.sign_out()
        except Exception:
            pass
    for key in list(st.session_state.keys()):   # forget everything about this user
        del st.session_state[key]


def _valid(email: str, password: str) -> str | None:
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        return "Please enter a valid email address."
    if len(password) < 6:
        return "Password must be at least 6 characters."
    return None


def login_page(app_name: str, tagline: str) -> None:
    """Draws the login / sign-up screen. Sets st.session_state['user'] on success."""
    st.markdown(f"<h1 style='text-align:center'>🔎 {app_name}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center;opacity:.75'>{tagline}</p>", unsafe_allow_html=True)
    if not supabase_enabled():
        st.info("Local mode: accounts are stored on this computer only.", icon="💻")

    _, middle, _ = st.columns([1, 3, 1])
    with middle:
        log_in_tab, sign_up_tab = st.tabs(["Log in", "Create account"])

        with log_in_tab, st.form("login"):
            email = st.text_input("Email").strip().lower()
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Log in", type="primary", use_container_width=True):
                try:
                    user = (_supabase_log_in if supabase_enabled() else _local_log_in)(email, password)
                    st.session_state["user"] = user
                    st.rerun()
                except Exception as e:
                    st.error(_friendly(e))

        with sign_up_tab, st.form("signup"):
            email = st.text_input("Email", key="su_email").strip().lower()
            password = st.text_input("Password (6+ characters)", type="password", key="su_pw")
            confirm = st.text_input("Repeat password", type="password", key="su_pw2")
            if st.form_submit_button("Create account", use_container_width=True):
                problem = _valid(email, password) or (None if password == confirm else "Passwords don't match.")
                if problem:
                    st.error(problem)
                else:
                    try:
                        user = (_supabase_sign_up if supabase_enabled() else _local_sign_up)(email, password)
                        if user is None:
                            st.success("Account created. Check your email to confirm it, then log in.")
                        else:
                            st.session_state["user"] = user
                            st.rerun()
                    except Exception as e:
                        st.error(_friendly(e))


def _friendly(e: Exception) -> str:
    msg = getattr(e, "message", None) or str(e)
    if "Invalid login credentials" in msg:
        return "Wrong email or password."
    if "Email not confirmed" in msg:
        return "Please confirm your email first (check your inbox), then log in."
    return msg