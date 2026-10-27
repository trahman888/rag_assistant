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
    st.session_state["user"] = {"id": "...", "email": "...", "name": "..."}
    display_name(user) -> the name to show (falls back to the email for old accounts)
    user["id"] names the user's private document folder (knowledge_base.Library).
    Supabase's storage rules check that same id, so one user can never read
    another user's files. Keep Supabase "Confirm email" ON.

STAYING LOGGED IN AFTER A PAGE REFRESH
    A refresh starts a brand-new Streamlit session, and st.session_state starts
    empty. So after login we put a login token in the page address:
        https://your-app.streamlit.app/?s=TOKEN
    A refresh keeps the address, and Streamlit reads it on the server
    (st.query_params), so this works on Streamlit Community Cloud, where
    cookies and in-page JavaScript can't be relied on.
      * Supabase mode: the token is the Supabase refresh token. Each page load
        exchanges it for a fresh session and puts the NEW token in the address,
        so an old copied address stops working.
      * Local mode: a signed token that expires after REMEMBER_DAYS.
    Log out removes it.

    Because the address contains the login, don't share the page link while
    logged in: share the plain app address instead.
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
URL_PARAM = "s"                     # ?s=TOKEN in the page address
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


# --------------------------------------------------------------------------- local mode
def _hash(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 200_000).hex()


def _load_users() -> dict:
    return json.loads(USERS_FILE.read_text()) if USERS_FILE.exists() else {}


def _save_users(users: dict) -> None:
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    USERS_FILE.write_text(json.dumps(users, indent=2))


def _local_sign_up(email: str, password: str, name: str) -> dict:
    users = _load_users()
    if email in users:
        raise ValueError("An account with this email already exists.")
    salt = secrets.token_hex(16)
    users[email] = {"id": secrets.token_hex(8), "name": name, "salt": salt, "hash": _hash(password, salt)}
    _save_users(users)
    return {"id": users[email]["id"], "email": email, "name": name}


def _local_log_in(email: str, password: str) -> dict:
    u = _load_users().get(email)
    if not u or not secrets.compare_digest(u["hash"], _hash(password, u["salt"])):
        raise ValueError("Wrong email or password.")
    return {"id": u["id"], "email": email, "name": u.get("name", "")}


# --------------------------------------------------------------------------- supabase mode
def _from_supabase(sb_user) -> dict:
    """Supabase keeps extra profile fields, like the name, in user_metadata."""
    meta = getattr(sb_user, "user_metadata", None) or {}
    return {"id": sb_user.id, "email": sb_user.email, "name": (meta.get("full_name") or "").strip()}


def _supabase_sign_up(email: str, password: str, name: str) -> dict | None:
    res = supabase_client().auth.sign_up(
        {"email": email, "password": password, "options": {"data": {"full_name": name}}})
    if res.session is None:   # project requires email confirmation
        return None
    _remember("sb." + res.session.refresh_token)
    return _from_supabase(res.user)


def _supabase_log_in(email: str, password: str) -> dict:
    res = supabase_client().auth.sign_in_with_password({"email": email, "password": password})
    _remember("sb." + res.session.refresh_token)
    return _from_supabase(res.user)


# --------------------------------------------------------------------------- forgot password / delete account
def request_password_reset(email: str) -> None:
    """Supabase emails a one-time code (the "Reset Password" email template must show
    {{ .Token }}; see SETUP_SUPABASE.md)."""
    supabase_client().auth.reset_password_for_email(email)


def reset_password_with_code(email: str, code: str, new_password: str) -> dict:
    """Check the emailed code, set the new password, and log the user in."""
    sb = supabase_client()
    res = sb.auth.verify_otp({"email": email, "token": code.strip(), "type": "recovery"})
    if not res.session:
        raise ValueError("That code didn't work. Check it, or ask for a new one.")
    sb.auth.update_user({"password": new_password})
    _remember("sb." + res.session.refresh_token)
    return _from_supabase(res.user)


def delete_account() -> None:
    """Delete the logged-in user's login. (Their files are deleted first by the app.)
    Supabase: calls the delete_user() database function from SETUP_SUPABASE.md, which can
    only ever delete the account of whoever is calling it. No secret key needed."""
    user = st.session_state["user"]
    if supabase_enabled():
        supabase_client().rpc("delete_user").execute()
    else:
        users = _load_users()
        users.pop(user["email"], None)
        _save_users(users)
    log_out()
    st.session_state["_account_deleted"] = True


# --------------------------------------------------------------------------- names
def display_name(user: dict) -> str:
    """The name to show. Accounts created before names existed fall back to the
    part of the email before the @ (tanzila.himu@… -> Tanzila Himu)."""
    if user.get("name"):
        return user["name"]
    return user["email"].split("@")[0].replace(".", " ").replace("_", " ").title()


def update_name(new_name: str) -> None:
    """Save a new display name for the logged-in user."""
    new_name = re.sub(r"\s+", " ", new_name).strip()[:60]
    if not new_name:
        raise ValueError("Please enter a name.")
    user = st.session_state["user"]
    if supabase_enabled():
        supabase_client().auth.update_user({"data": {"full_name": new_name}})
    else:
        users = _load_users()
        users[user["email"]]["name"] = new_name
        _save_users(users)
    user["name"] = new_name


# --------------------------------------------------------------------------- "remember me" (page address)
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
        if data["exp"] <= time.time():
            return None
        name = _load_users().get(data["email"], {}).get("name", "")    # always the latest name
        return {"id": data["id"], "email": data["email"], "name": name}
    except Exception:
        return None


def _remember(token: str) -> None:
    """Put the login token in the page address (…/?s=TOKEN). Streamlit keeps it
    through a refresh and reads it on the server, with no browser code needed."""
    st.query_params[URL_PARAM] = token


def _forget() -> None:
    if URL_PARAM in st.query_params:
        del st.query_params[URL_PARAM]


def restore_login(token: str) -> None:
    """After a refresh: log the person back in from their saved token, if it's still valid."""
    user = None
    if token.startswith("sb.") and supabase_enabled():
        try:
            res = supabase_client().auth.refresh_session(token[3:])
            if res.session and res.user:
                user = _from_supabase(res.user)
                _remember("sb." + res.session.refresh_token)     # Supabase rotated it; save the new one
        except Exception:
            user = None
    elif token.startswith("lo.") and not supabase_enabled():
        user = _read_local_token(token)
    if user:
        st.session_state["user"] = user
    else:
        _forget()                                                # expired or invalid: drop it
        st.session_state["_restore_failed"] = True


def _logged_in(user: dict) -> None:
    st.session_state["user"] = user
    st.session_state["_fresh_login"] = True    # typed their password: start with a new chat (chatapp.py)
    st.session_state.pop("_logged_out", None)
    st.session_state.pop("_restore_failed", None)
    if not supabase_enabled():                 # Supabase mode already saved its refresh token
        _remember(_local_token(user))


# --------------------------------------------------------------------------- public
def current_user() -> dict | None:
    """The logged-in user, restoring it from the page address after a refresh."""
    if "user" in st.session_state:
        return st.session_state["user"]
    if st.session_state.get("_logged_out") or st.session_state.get("_restore_failed"):
        return None
    token = st.query_params.get(URL_PARAM, "")
    if token:
        restore_login(token)
    return st.session_state.get("user")


def log_out() -> None:
    if supabase_enabled() and "sb" in st.session_state:
        try:
            supabase_client().auth.sign_out()
        except Exception:
            pass
    for key in list(st.session_state.keys()):   # forget everything about this user
        del st.session_state[key]
    st.session_state["_logged_out"] = True      # don't auto-restore the old login in this tab
    _forget()                                   # remove the token from the page address


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

    if st.session_state.pop("_account_deleted", False):
        st.success("Your account and all your data have been deleted.", icon="🗑️")

    _, middle, _ = st.columns([1, 3, 1])
    with middle:
        log_in_tab, sign_up_tab, forgot_tab = st.tabs(["Log in", "Create account", "Forgot password"])

        with forgot_tab:
            _forgot_password_form()

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
            name = re.sub(r"\s+", " ", st.text_input("Full name", key="su_name", max_chars=60)).strip()
            email = st.text_input("Email", key="su_email").strip().lower()
            password = st.text_input("Password (6+ characters)", type="password", key="su_pw")
            confirm = st.text_input("Repeat password", type="password", key="su_pw2")
            if st.form_submit_button("Create account", use_container_width=True):
                problem = (("Please enter your name." if not name else None)
                           or _valid(email, password)
                           or (None if password == confirm else "Passwords don't match."))
                if problem:
                    st.error(problem)
                else:
                    try:
                        user = (_supabase_sign_up if supabase_enabled() else _local_sign_up)(email, password, name)
                        if user is None:
                            st.success("Account created. Check your email to confirm it, then log in.")
                        else:
                            _logged_in(user)
                            st.rerun()
                    except Exception as e:
                        st.error(_friendly(e))


def _forgot_password_form() -> None:
    """Step 1: email me a code.  Step 2: code + new password."""
    if not supabase_enabled():
        st.info("Local mode has no email, so passwords can't be reset here. "
                "Create a new account, or delete user_data/users.json to start over.")
        return
    sent_to = st.session_state.get("_reset_email")
    if not sent_to:
        with st.form("forgot"):
            email = st.text_input("Your account's email", key="fp_email").strip().lower()
            if st.form_submit_button("Email me a reset code", use_container_width=True):
                if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
                    st.error("Please enter a valid email address.")
                else:
                    try:
                        request_password_reset(email)
                        st.session_state["_reset_email"] = email
                        st.rerun()
                    except Exception as e:
                        st.error(_friendly(e))
        return

    st.caption(f"If an account exists for **{sent_to}**, we've emailed it a reset code. "
               "It can take a minute; check your spam folder too.")
    with st.form("reset"):
        code = st.text_input("Code from the email", key="fp_code", max_chars=10)
        new_pw = st.text_input("New password (6+ characters)", type="password", key="fp_pw")
        confirm = st.text_input("Repeat new password", type="password", key="fp_pw2")
        if st.form_submit_button("Set new password and log in", type="primary", use_container_width=True):
            problem = ((None if code.strip().isdigit() else "Enter the number from the email.")
                       or _valid(sent_to, new_pw)
                       or (None if new_pw == confirm else "Passwords don't match."))
            if problem:
                st.error(problem)
            else:
                try:
                    user = reset_password_with_code(sent_to, code, new_pw)
                    st.session_state.pop("_reset_email", None)
                    _logged_in(user)
                    st.rerun()
                except Exception as e:
                    st.error(_friendly(e))
    if st.button("Use a different email / send a new code", type="tertiary"):
        st.session_state.pop("_reset_email", None)
        st.rerun()


def _friendly(e: Exception) -> str:
    msg = getattr(e, "message", None) or str(e)
    if "Invalid login credentials" in msg:
        return "Wrong email or password."
    if "Email not confirmed" in msg:
        return "Please confirm your email first (check your inbox), then log in."
    if "expired" in msg.lower() or "invalid" in msg.lower() and "token" in msg.lower():
        return "That code is wrong or has expired. Ask for a new one."
    if "rate limit" in msg.lower() or "429" in msg:
        return "Too many emails were sent recently. Please wait a few minutes and try again."
    if "should be different" in msg.lower():
        return "Choose a password different from your old one."
    return msg
