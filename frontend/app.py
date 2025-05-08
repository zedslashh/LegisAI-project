import streamlit as st
import requests
import os
from datetime import datetime
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import base64

API_URL = "http://localhost:8000"

st.set_page_config(layout="wide")

# Sidebar for navigation (moved to left)
with st.sidebar:
    selected = option_menu(
        "Navigation",
        ["Upload", "Categories", "Search", "Timeline", "Upcoming Cases", "Case Features"],
        icons=['upload', 'categories', 'search', 'timeline', 'upcoming', 'list-task'],
        default_index=0,
        orientation="vertical"
    )

def display_upload_tab():
    st.header("Upload a Case PDF")
    st.subheader("Summarization Options")
    summary_style = st.radio(
        "Choose summary style:",
        ["default", "brief", "detailed", "structured"],
        format_func=lambda x: {
            "default": "Standard (5 sentences)",
            "brief": "Brief (3 sentences)",
            "detailed": "Detailed (8-10 sentences)",
            "structured": "Structured (with sections)"
        }[x]
    )
    style_descriptions = {
        "default": "A balanced 5-sentence summary covering the main points of the case.",
        "brief": "A concise 3-sentence summary focusing on the main issue and outcome.",
        "detailed": "A comprehensive summary (8-10 sentences) including key facts, legal issues, arguments, decision, and implications.",
        "structured": "A structured summary with sections: Case Overview, Key Facts, Legal Issues, Decision, and Impact."
    }
    st.info(style_descriptions[summary_style])
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
    if uploaded_file:
        with st.spinner("Processing..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                data = {"summary_style": summary_style}
                resp = requests.post(f"{API_URL}/upload/", files=files, data=data)
                if resp.ok:
                    data = resp.json()
                    if "error" in data:
                        st.error(f"Error: {data['error']}")
                    else:
                        st.success("Uploaded and processed!")
                        if "metadata" in data:
                            st.subheader("Case Information")
                            metadata = data["metadata"]
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Case Number:**", metadata["case_number"])
                                st.write("**Filing Date:**", metadata["filing_date"])
                                st.write("**Court:**", metadata["court"])
                                st.write("**Judge:**", metadata["judge"])
                            with col2:
                                st.write("**Status:**", metadata["status"])
                                if metadata["next_hearing"]:
                                    st.write("**Next Hearing:**", metadata["next_hearing"])
                            if metadata["parties"]:
                                st.write("**Parties Involved:**")
                                for party in metadata["parties"]:
                                    st.write(f"- {party}")
                            if metadata["related_cases"]:
                                st.write("**Related Cases:**")
                                for case in metadata["related_cases"]:
                                    st.write(f"- {case}")
                        if "summary" in data:
                            st.subheader("Generated Summary")
                            st.write(data["summary"])
                        if "category" in data:
                            st.write("**Category:**", data["category"])
                        if "filename" in data:
                            st.write("**Saved as:**", data["filename"])
                else:
                    error_msg = resp.json().get("error", "Upload failed")
                    st.error(f"Upload failed: {error_msg}")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

def display_search_tab():
    st.header("Semantic Search & Q&A (RAG)")
    query = st.text_input("Ask a question about the case (e.g., 'Who is the judge?')")
    if query:
        try:
            # Call a new backend endpoint for RAG-based QA (to be implemented in backend)
            resp = requests.get(f"{API_URL}/rag_qa/", params={"query": query})
            if resp.ok:
                answer = resp.json().get("answer", "No answer found.")
                st.success(f"**Answer:** {answer}")
                # Optionally show supporting context
                if "context" in resp.json():
                    st.write("**Supporting Context:**")
                    st.write(resp.json()["context"])
            else:
                error_msg = resp.json().get("error", "Search failed")
                st.error(f"Search failed: {error_msg}")
        except Exception as e:
            st.error(f"Error performing search: {str(e)}")

def display_case_features_tab():
    st.header("Case Features")
    st.write("Extracted features from the uploaded case documents:")
    try:
        resp = requests.get(f"{API_URL}/case_features/")
        if resp.ok:
            features = resp.json()
            for idx, feature in enumerate(features):
                with st.expander(f"Case {idx+1}"):
                    for k, v in feature.items():
                        st.write(f"**{k.replace('_', ' ').title()}:** {v}")
        else:
            st.error("Failed to fetch case features")
    except Exception as e:
        st.error(f"Error loading case features: {str(e)}")

def display_categories_tab():
    st.header("Document Categories")
    try:
        resp = requests.get(f"{API_URL}/categories/")
        if resp.ok:
            categories = resp.json()
            cols = st.columns(3)
            for idx, category in enumerate(categories):
                with cols[idx % 3]:
                    st.subheader(f"{category['name']} ({category['count']} files)")
                    cat_resp = requests.get(f"{API_URL}/category/{category['name']}")
                    if cat_resp.ok:
                        cat_data = cat_resp.json()
                        if "files" in cat_data:
                            for file in cat_data["files"]:
                                with st.expander(file):
                                    st.write(f"Category: {category['name']}")
        else:
            st.error("Failed to fetch categories")
    except Exception as e:
        st.error(f"Error loading categories: {str(e)}")

def display_timeline_tab():
    st.header("Case Timeline")
    try:
        resp = requests.get(f"{API_URL}/cases/timeline")
        if resp.ok:
            timeline = resp.json()
            if not timeline:
                st.info("No cases in timeline")
            else:
                for case in timeline:
                    with st.expander(f"{case['case_number']} - {case['filing_date']}"):
                        st.write(f"**Court:** {case['court']}")
                        st.write(f"**Status:** {case['status']}")
                        st.write(f"**Category:** {case['category']}")
                        if case['next_hearing']:
                            st.write(f"**Next Hearing:** {case['next_hearing']}")
        else:
            st.error("Failed to fetch timeline")
    except Exception as e:
        st.error(f"Error loading timeline: {str(e)}")

def display_upcoming_cases_tab():
    st.header("Upcoming Cases")
    try:
        resp = requests.get(f"{API_URL}/cases/upcoming")
        if resp.ok:
            upcoming = resp.json()
            if not upcoming:
                st.info("No upcoming cases")
            else:
                for case in upcoming:
                    with st.expander(f"{case['case_number']} - {case['next_hearing']}"):
                        st.write(f"**Court:** {case['court']}")
                        st.write(f"**Category:** {case['category']}")
        else:
            st.error("Failed to fetch upcoming cases")
    except Exception as e:
        st.error(f"Error loading upcoming cases: {str(e)}")

# Main navigation logic
if selected == "Upload":
    display_upload_tab()
elif selected == "Categories":
    display_categories_tab()
elif selected == "Search":
    display_search_tab()
elif selected == "Timeline":
    display_timeline_tab()
elif selected == "Upcoming Cases":
    display_upcoming_cases_tab()
elif selected == "Case Features":
    display_case_features_tab()
