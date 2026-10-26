"""
The 📊 Evaluation page: feedback summary, test set, runs and results.
Everything here belongs to the logged-in user and uses only their documents.
"""
import pandas as pd
import streamlit as st

from evaluation import (EVAL_FILE, MAX_RUNS, MODES, RUNS_FILE, checkable, clean_case, evaluate,
                        generate_cases)
from knowledge_base import FEEDBACK_FILE

COLUMNS = ["question", "expected_text", "document", "page", "in_documents", "origin"]


def _pct(x) -> str:
    return "–" if x is None else f"{x * 100:.0f}%"


def _latest_feedback(items: list[dict]) -> list[dict]:
    """One rating per answer (the user may have changed their mind)."""
    by_id = {}
    for f in items:
        by_id[f.get("answer_id") or id(f)] = f
    return [f for f in by_id.values() if f.get("rating") in ("up", "down")]


def render(library, settings: dict, llm_complete, plan_fn, answer_fn) -> None:
    st.title("📊 Evaluation")
    st.caption("Measure how well the assistant finds and cites the right passages in **your** documents. "
               "Results are private to you.")

    docs = [d["source"] for d in library.documents()]
    cases = [c for c in (clean_case(c) for c in library.load_json(EVAL_FILE, [])) if c]
    feedback = _latest_feedback(library.load_json(FEEDBACK_FILE, []))

    # ------------------------------------------------------------------ 1. feedback
    st.subheader("👍 👎 Feedback on answers")
    up = sum(f["rating"] == "up" for f in feedback)
    down = len(feedback) - up
    a, b, c = st.columns(3)
    a.metric("👍 Helpful", up)
    b.metric("👎 Not helpful", down)
    c.metric("Satisfaction", _pct(up / len(feedback)) if feedback else "–")
    bad = [f for f in reversed(feedback) if f["rating"] == "down"]
    if bad:
        with st.expander(f"Review 👎 answers ({len(bad)})"):
            st.caption("Turn a bad answer into a test question, so you can check whether a change fixes it.")
            known = {x["question"] for x in cases}
            for i, f in enumerate(bad[:20]):
                st.markdown(f"**Q:** {f['question']}")
                st.caption(f"A: {f['answer'][:300]}")
                cited = [s for s in f.get("sources", []) if s.get("cited")]
                if f["question"] in known:
                    st.caption("✔️ already in the test set")
                elif st.button("➕ Add to test set", key=f"addfb-{i}"):
                    cases.append({"question": f["question"], "expected_text": "",
                                  "document": cited[0]["source"] if cited else "", "page": None,
                                  "in_documents": True, "origin": "👎 feedback"})
                    library.save_json(EVAL_FILE, cases)
                    st.toast("Added. Fill in the expected text below.", icon="➕")
                    st.rerun()
                st.divider()
    elif not feedback:
        st.caption("No ratings yet. Use 👍 / 👎 under answers in the chat.")

    # ------------------------------------------------------------------ 2. test set
    st.subheader("🧪 Test questions")
    st.caption("For each question, put a short phrase the **correct passage contains** in *Expected text* "
               "(copy 4–12 words from the document). Untick *In documents* for questions your documents "
               "can't answer: the right behaviour then is \"I couldn't find this\".")
    df = pd.DataFrame(cases or [], columns=COLUMNS)
    edited = st.data_editor(
        df, num_rows="dynamic", use_container_width=True, hide_index=True, key="eval_editor",
        column_config={
            "question": st.column_config.TextColumn("Question", width="large", required=True),
            "expected_text": st.column_config.TextColumn("Expected text", width="medium",
                                                         help="A phrase copied from the correct passage"),
            "document": st.column_config.SelectboxColumn("Document", options=[""] + docs,
                                                         help="Optional: the file the answer is in"),
            "page": st.column_config.NumberColumn("Page", min_value=1, step=1, help="Optional"),
            "in_documents": st.column_config.CheckboxColumn("In documents", default=True),
            "origin": st.column_config.TextColumn("Origin", disabled=True, default="manual"),
        })
    new_cases = [c for c in (clean_case(r) for r in edited.to_dict("records")) if c]
    unchecked = [c["question"] for c in new_cases if not checkable(c)]

    s1, s2, s3 = st.columns([2, 2, 3], vertical_alignment="bottom")
    if s1.button("💾 Save test set", use_container_width=True, disabled=new_cases == cases):
        library.save_json(EVAL_FILE, new_cases)
        st.toast(f"Saved {len(new_cases)} test questions", icon="💾")
        st.rerun()
    how_many = s3.number_input("Questions to generate", 1, 10, 5, label_visibility="collapsed")
    if s2.button("✨ Generate from my documents", use_container_width=True, disabled=not docs,
                 help="The AI picks random passages from your documents and writes a question for each "
                      "(one AI call per question). Check them: generated questions tend to be easier than real ones."):
        with st.spinner("Writing test questions…"):
            made = generate_cases(library, llm_complete, int(how_many))
        if made:
            library.save_json(EVAL_FILE, new_cases + made)
            st.toast(f"Added {len(made)} questions", icon="✨")
            st.rerun()
        else:
            st.warning("Couldn't generate questions right now. Try again in a minute.")
    if unchecked:
        st.caption(f"⚠️ {len(unchecked)} question(s) have no expected text or document yet and will be skipped.")

    # ------------------------------------------------------------------ 3. run
    st.subheader("▶️ Run evaluation")
    k = settings["top_k"]
    modes = st.multiselect("Compare these search methods", list(MODES), default=list(MODES),
                           format_func=MODES.get)
    check_answers = st.checkbox("Also check the answers (one AI call per question, uses your free quota)")
    usable = [c for c in new_cases if checkable(c)]
    if st.button(f"Run on {len(usable)} question(s) with Top K = {k}", type="primary",
                 disabled=not usable or not modes or new_cases != cases,
                 help="Save the test set first" if new_cases != cases else None):
        bar = st.progress(0.0, text="Starting…")
        try:
            report = evaluate(library, usable, modes, k, plan_fn=plan_fn if "rewrite" in modes or check_answers else None,
                              answer_fn=answer_fn if check_answers else None,
                              progress=lambda f, label: bar.progress(min(f, 1.0), text=label[:90]))
            runs = library.load_json(RUNS_FILE, [])
            runs.append(report)
            library.save_json(RUNS_FILE, runs[-MAX_RUNS:])
        except Exception as e:
            st.error(f"Evaluation stopped: {e}")
        bar.empty()

    # ------------------------------------------------------------------ 4. results
    runs = library.load_json(RUNS_FILE, [])
    if not runs:
        st.caption("No runs yet.")
        return
    last = runs[-1]
    st.subheader("📈 Results")
    st.caption(f"Latest run: {last['n_cases']} question(s), Top K = {last['k']}, "
               f"{last['time'].replace('T', ' ')[:16]} UTC")

    table = pd.DataFrame([{"Search method": MODES[m], f"Hit rate @{last['k']}": r["hit_rate"] * 100,
                           "MRR": r["mrr"]} for m, r in last["modes"].items()])
    st.dataframe(table, hide_index=True, use_container_width=True, column_config={
        f"Hit rate @{last['k']}": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
        "MRR": st.column_config.NumberColumn(format="%.2f")})
    st.caption(f"**Hit rate @{last['k']}**: share of questions where a correct passage was retrieved. "
               "**MRR**: how high it ranked (1.00 = always first, 0.50 = second on average).")

    ans = last.get("answers")
    if ans:
        a, b = st.columns(2)
        a.metric("Answers citing a correct passage", _pct(ans["cited_rate"]), help=f"{ans['n_cited']} question(s)")
        b.metric("Correct \"not found\"", _pct(ans["not_found_rate"]), help=f"{ans['n_not_found']} question(s)")
        if ans.get("errors"):
            st.caption(f"⚠️ {ans['errors']} answer(s) failed (usually the AI's free-tier limit). Run again later.")

    with st.expander("Per-question details"):
        detail = {}
        for m, r in last["modes"].items():
            for row in r["rows"]:
                detail.setdefault(row["question"], {"Question": row["question"]})[MODES[m]] = \
                    f"#{row['rank']}" if row["rank"] else "✗ missed"
        if ans:
            for row in ans["rows"]:
                detail.setdefault(row["question"], {"Question": row["question"]})["Answer"] = row["verdict"]
        st.dataframe(pd.DataFrame(list(detail.values())), hide_index=True, use_container_width=True)
        st.caption("#1 = the correct passage came first; ✗ = not in the top K.")

    if len(runs) > 1:
        st.markdown("**Hit rate over time**")
        hist = pd.DataFrame([{"run": i + 1, **{MODES[m]: r["hit_rate"] * 100 for m, r in run["modes"].items()}}
                             for i, run in enumerate(runs)]).set_index("run")
        st.line_chart(hist, y_label="Hit rate (%)", x_label="Run")
