# -----------------------------
# One-click example prompts (shown BELOW the textbox)
# -----------------------------
with st.expander("View example prompts", expanded=False):  # set True if you want it opened by default
    cols = st.columns(2)
    picked = None
    for i, p in enumerate(EXAMPLE_PROMPTS):
        if cols[i % 2].button(p, key=f"ex_{i}"):
            picked = p

    if picked is not None:
        # stage the query, then rerun (this is NOT inside a callback)
        st.session_state["pending_query"] = picked
        st.session_state["auto_submit"] = True
        st.rerun()
