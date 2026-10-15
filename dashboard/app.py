"""Feeder dashboard: counts, recent visits, and a review queue.

    streamlit run dashboard/app.py
    streamlit run dashboard/app.py -- --set paths.db=data/sim/feeder.db --set paths.incoming=data/sim/incoming
"""

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from feedercam import db  # noqa: E402
from feedercam.config import load_config, resolve  # noqa: E402

_args = argparse.ArgumentParser()
_args.add_argument("--config")
_args.add_argument("--set", action="append", default=[])
_args = _args.parse_args()
cfg = load_config(_args.config, _args.set)
INCOMING = resolve(cfg["paths"]["incoming"])
LOW_CONF = cfg["classifier"]["low_conf"]
EXTRA_LABELS = ["Not a bird", "Unknown"]

st.set_page_config(page_title="Feeder cam", page_icon="🐦", layout="wide")


# A fresh connection each rerun: SQLite connections can't be shared across
# the threads Streamlit uses for different reruns.
conn = db.connect(resolve(cfg["paths"]["db"]))


def species_options():
    path = ROOT / "training" / "species_nc.txt"
    names = [l.strip() for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")]
    seen = [r[0] for r in conn.execute("SELECT DISTINCT species FROM visits WHERE species IS NOT NULL")]
    return sorted(set(names) | set(seen)) + EXTRA_LABELS


@st.cache_data(ttl=30)
def load_visits(start: date, end: date) -> pd.DataFrame:
    df = pd.read_sql_query(
        """SELECT v.*, (SELECT path FROM crops c WHERE c.visit_uid = v.uid ORDER BY id LIMIT 1) AS crop
           FROM labeled_visits v WHERE day BETWEEN ? AND ? ORDER BY started_at DESC""",
        conn, params=(start.isoformat(), end.isoformat()),
    )
    df["started_at"] = pd.to_datetime(df["started_at"])
    df["ended_at"] = pd.to_datetime(df["ended_at"])
    df["duration_s"] = (df["ended_at"] - df["started_at"]).dt.total_seconds().round(1)
    return df


def bar_color():
    # One series per chart, so one hue: the reference palette's slot 1,
    # stepped for whichever theme Streamlit is showing.
    dark = getattr(getattr(st.context, "theme", None), "type", "light") == "dark"
    return "#3987e5" if dark else "#2a78d6"


def bar_chart(df, x, y, horizontal=False, x_title=None, y_title=None, tooltip=None):
    mark = alt.Chart(df).mark_bar(
        color=bar_color(), size=16,
        cornerRadiusEnd=4,  # rounded at the data end, square at the baseline
    )
    if horizontal:
        enc = dict(x=alt.X(f"{x}:Q", title=x_title), y=alt.Y(f"{y}:N", sort="-x", title=None))
    else:
        enc = dict(x=alt.X(f"{x}", title=x_title), y=alt.Y(f"{y}:Q", title=y_title))
    return mark.encode(**enc, tooltip=tooltip or [x, y])


def show_crop(rel_path, caption=None):
    path = INCOMING / rel_path if rel_path else None
    if path and path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.caption("(image not synced yet)")


def label_text(row):
    if row["corrected_species"]:
        return f"{row['corrected_species']} ✓"
    if row["species"]:
        return f"{row['species']} ({row['species_conf']:.0%})"
    return "not classified yet"


# --- sidebar ------------------------------------------------------------------

st.sidebar.title("🐦 Feeder cam")
today = date.today()
picked = st.sidebar.date_input("Dates", (today - timedelta(days=6), today), max_value=today)
start, end = picked if len(picked) == 2 else (picked[0], picked[0])  # mid-selection gives one date
if st.sidebar.button("Refresh"):
    st.cache_data.clear()
st.sidebar.caption(
    "New visits appear after `python -m feedercam ingest` and `classify`. "
    f"Predictions under {LOW_CONF:.0%} confidence go to the review queue."
)

visits = load_visits(start, end)
labeled = visits.dropna(subset=["final_species"])
labeled = labeled[~labeled["final_species"].isin(EXTRA_LABELS)]
review = visits[(visits["corrected_species"].isna()) &
                (visits["species"].notna()) & (visits["species_conf"] < LOW_CONF)]

overview, recent, review_tab = st.tabs(["Overview", "Recent visits", f"Review ({len(review)})"])

# --- overview -----------------------------------------------------------------

with overview:
    if visits.empty:
        st.info("No visits in this date range yet. Run the capture loop, then "
                "`python -m feedercam ingest`.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Visits", len(visits))
        c2.metric("Species seen", labeled["final_species"].nunique())
        c3.metric("Waiting for review", len(review))
        c4.metric("Not classified yet", int(visits["species"].isna().sum()))

        left, right = st.columns([3, 2])
        with left:
            st.subheader("Visits by species")
            by_species = labeled.groupby("final_species").size().reset_index(name="visits")
            if by_species.empty:
                st.caption("No classified visits yet.")
            else:
                height = max(120, 26 * len(by_species))
                st.altair_chart(
                    bar_chart(by_species, "visits", "final_species", horizontal=True,
                              x_title="Visits").properties(height=height),
                    width="stretch",
                )
        with right:
            st.subheader("Visits per day")
            per_day = visits.groupby("day").size().reset_index(name="visits")
            st.altair_chart(
                bar_chart(per_day, "day:O", "visits", x_title=None, y_title="Visits")
                .properties(height=220), width="stretch",
            )
            st.subheader("Time of day")
            hours = visits["started_at"].dt.hour.value_counts().reindex(range(24), fill_value=0)
            per_hour = hours.rename_axis("hour").reset_index(name="visits")
            st.altair_chart(
                bar_chart(per_hour, "hour:O", "visits", x_title="Hour", y_title="Visits")
                .properties(height=200), width="stretch",
            )

        with st.expander("Table view"):
            if not by_species.empty:
                st.dataframe(by_species.sort_values("visits", ascending=False),
                             hide_index=True, width="stretch")

# --- recent visits ------------------------------------------------------------

with recent:
    shown = visits.head(24)
    if shown.empty:
        st.info("Nothing here yet.")
    for chunk_start in range(0, len(shown), 6):
        cols = st.columns(6)
        for col, (_, row) in zip(cols, shown.iloc[chunk_start:chunk_start + 6].iterrows()):
            with col:
                show_crop(row["crop"])
                st.markdown(f"**{label_text(row)}**")
                st.caption(f"{row['started_at']:%b %d %H:%M:%S} · {row['duration_s']}s")

# --- review queue -------------------------------------------------------------

with review_tab:
    st.write("Low-confidence predictions. Fixing these builds training data "
             "from your own feeder; run `python -m feedercam export` and retrain to use it.")
    options = species_options()
    if review.empty:
        st.success("Nothing to review.")
    for _, row in review.head(15).iterrows():
        with st.container(border=True):
            crops = db.crops_for(conn, row["uid"])
            cols = st.columns([1, 1, 1, 2])
            for col, c in zip(cols[:3], crops):
                with col:
                    show_crop(c["path"])
            with cols[3]:
                top3 = json.loads(row["top3"]) if row["top3"] else []
                st.markdown(f"**{row['started_at']:%b %d %H:%M:%S}** · {row['duration_s']}s")
                st.caption("Model's guesses: " + ", ".join(f"{s} {p:.0%}" for s, p in top3))
                default = options.index(row["species"]) if row["species"] in options else 0
                choice = st.selectbox("Correct species", options, index=default, key=f"sel-{row['uid']}")
                if st.button("Save", key=f"save-{row['uid']}"):
                    db.set_correction(conn, row["uid"], choice)
                    st.cache_data.clear()
                    st.rerun()
