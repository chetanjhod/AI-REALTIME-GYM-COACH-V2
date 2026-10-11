import streamlit as st
from services.persistence.exercise_repository import get_or_create_user


def render_login_wall():
    if st.session_state.get("username") is not None:
        return True

    st.title("🏋️AI real time gym Trainer")
    st.markdown("### welcome! plese enter the username to start.")

    with st.form("login_form", clear_on_submit=False):
        username = st.text_input(
            "name (unique)",
            placeholder="uniq name e. g. chetanjhod"
        )

        submit_button = st.form_submit_button(
            "start session 🚀",
            width="stretch"
        )

    if submit_button:
        if not username.strip():
            st.error("name can not be empty.")
            return False

        user = get_or_create_user(username.strip())
        st.session_state["user_id"] = user["id"]
        st.session_state["username"] = user["username"]
        st.rerun()

    st.markdown(
        """
        <div style="
            text-align: center;
            padding: 20px 10px;
            margin-top: 30px;
            border-top: 1px solid rgba(255,255,255,0.12);
            color: #B0B7C3;
            font-size: 14px;
            letter-spacing: 1px;
        ">
            Created with ❤️ by
            <span style="
                color: #FF5CA8;
                font-weight: 700;
            ">
                JHOD SAHAB
            </span>
            <br>
            <span style="
                display: inline-block;
                margin-top: 8px;
                font-size: 11px;
                letter-spacing: 2px;
                color: #B0B7C3;
            ">
                AI REAL-TIME GYM COACH
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    return False