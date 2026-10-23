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
    is_admin(user) -> True for emails listed in ADMIN_EMAILS (default below).

ADMIN SAFETY
    Admin rights come from the email address, so make sure the admin account
    exists before anyone else can sign up with that address: create it first,
    and keep Supabase "Confirm email" ON so nobody can claim an email they
    don't own.

STAYING LOGGED IN AFTER A PAGE REFRESH
    A refresh starts a brand-new Streamlit session, and st.session_state starts
    empty. So after login we save a token in a browser cookie ("pa_session"),
    and on the next page load restore_login() reads it back:
      * Supabase mode: the cookie holds the Supabase refresh token. We exchange
        it for a fresh session (Supabase also hands back a new refresh token,
        which we save, so the old one stops working).
      * Local mode: the cookie holds a signed, expiring token.
    Log out deletes the cookie. Tokens last REMEMBER_DAYS.
"""
import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import time
from pathlib import Path

import streamlit as st

USERS_FILE = Path(__file__).parent / "user_data" / "users.json"
SECRET_FILE = Path(__file__).parent / "user_data" / ".session_secret"
COOKIE = "pa_session"
REMEMBER_DAYS = 14


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
        from urllib.parse import urlparse

        from supabase import create_client
        u = urlparse(_secret("SUPABASE_URL").strip())
        base_url = f"{u.scheme}://{u.netloc}"            # tolerate /rest/v1/ or a trailing slash
        st.session_state["sb"] = create_client(base_url, _secret("SUPABASE_KEY").strip())
    return st.session_state["sb"]


# --------------------------------------------------------------------------- admin
DEFAULT_ADMINS = "tanzila.himu@gmail.com"


def admin_emails() -> set[str]:
    """Admins come from the ADMIN_EMAILS secret (comma-separated), else the default above."""
    raw = _secret("ADMIN_EMAILS") or DEFAULT_ADMINS
    return {e.strip().lower() for e in raw.split(",") if e.strip()}


def is_admin(user: dict | None) -> bool:
    return bool(user) and user["email"].lower() in admin_emails()


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
    _remember("sb." + res.session.refresh_token)
    return {"id": res.user.id, "email": res.user.email}


def _supabase_log_in(email: str, password: str) -> dict:
    res = supabase_client().auth.sign_in_with_password({"email": email, "password": password})
    _remember("sb." + res.session.refresh_token)
    return {"id": res.user.id, "email": res.user.email}


# --------------------------------------------------------------------------- "remember me" cookie
def _server_secret() -> bytes:
    """Key for signing local-mode tokens (secret SESSION_SECRET, else a random file)."""
    s = _secret("SESSION_SECRET")
    if s:
        return s.encode()
    if not SECRET_FILE.exists():
        SECRET_FILE.parent.mkdir(parents=True, exist_ok=True)
        SECRET_FILE.write_text(secrets.token_hex(32))
    return SECRET_FILE.read_text().encode()


def _local_token(user: dict) -> str:
    payload = base64.urlsafe_b64encode(json.dumps(
        {"id": user["id"], "email": user["email"], "exp": int(time.time()) + REMEMBER_DAYS * 86400}
    ).encode()).decode().rstrip("=")
    sig = hmac.new(_server_secret(), payload.encode(), hashlib.sha256).hexdigest()
    return f"lo.{payload}.{sig}"


def _read_local_token(token: str) -> dict | None:
    try:
        _, payload, sig = token.split(".")
        good = hmac.new(_server_secret(), payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, good):
            return None
        data = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        return {"id": data["id"], "email": data["email"]} if data["exp"] > time.time() else None
    except Exception:
        return None


def _remember(token: str) -> None:
    st.session_state["_cookie_write"] = token        # written to the browser on the next run


def _cookie_script(value: str, max_age: int) -> None:
    secure = "; Secure" if str(st.context.url or "").startswith("https") else ""
    st.html(f"<script>document.cookie = '{COOKIE}={value}; path=/; max-age={max_age}; SameSite=Lax{secure}';</script>",
            unsafe_allow_javascript=True)


def sync_cookie() -> None:
    """Call once per run, early. Writes or deletes the browser cookie when needed."""
    token = st.session_state.pop("_cookie_write", None)
    if token:
        _cookie_script(token, REMEMBER_DAYS * 86400)
    if st.session_state.pop("_cookie_clear", False):
        _cookie_script("", 0)


def restore_login() -> None:
    """After a refresh: log the person back in from their cookie, if it's still valid."""
    if "user" in st.session_state or st.session_state.get("_logged_out"):
        return
    token = (st.context.cookies or {}).get(COOKIE, "")
    if not token:
        return
    user = None
    if token.startswith("sb.") and supabase_enabled():
        try:
            res = supabase_client().auth.refresh_session(token[3:])
            if res.session and res.user:
                user = {"id": res.user.id, "email": res.user.email}
                _remember("sb." + res.session.refresh_token)     # Supabase rotated it; save the new one
        except Exception:
            user = None
    elif token.startswith("lo.") and not supabase_enabled():
        user = _read_local_token(token)
    if user:
        st.session_state["user"] = user
    else:
        st.session_state["_cookie_clear"] = True                 # expired or invalid: forget it


def _logged_in(user: dict) -> None:
    st.session_state["user"] = user
    st.session_state.pop("_logged_out", None)
    if not supabase_enabled():                 # Supabase mode already saved its refresh token
        _remember(_local_token(user))


# --------------------------------------------------------------------------- public
def current_user() -> dict | None:
    restore_login()
    return st.session_state.get("user")


def log_out() -> None:
    if supabase_enabled() and "sb" in st.session_state:
        try:
            supabase_client().auth.sign_out()
        except Exception:
            pass
    for key in list(st.session_state.keys()):   # forget everything about this user
        del st.session_state[key]
    st.session_state["_logged_out"] = True      # don't auto-restore from the old cookie in this tab
    st.session_state["_cookie_clear"] = True    # delete the cookie on the next run


def _valid(email: str, password: str) -> str | None:
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        return "Please enter a valid email address."
    if len(password) < 6:
        return "Password must be at least 6 characters."
    return None


def login_page(app_name: str, tagline: str) -> None:
    """Draws the login / sign-up screen. Sets st.session_state['user'] on success."""
    if app_name:
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
                    _logged_in(user)
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
                            _logged_in(user)
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
