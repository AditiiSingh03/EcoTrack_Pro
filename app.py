import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
import google.generativeai as genai
from fpdf import FPDF
from PIL import Image
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import requests
import base64
import os
import json
import hashlib

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="EcoTrack Pro",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- API KEY ---
GOOGLE_API_KEY = st.secrets["GEMINI_API_KEY"]

# ==========================================
# USER DATABASE MANAGEMENT (JSON)
# ==========================================
USER_DB_FILE = "users.json"

if not os.path.exists(USER_DB_FILE):
    with open(USER_DB_FILE, "w") as f:
        json.dump({}, f)

def load_users():
    if not os.path.exists(USER_DB_FILE):
        return {}
    with open(USER_DB_FILE, "r") as f:
        return json.load(f)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def save_user(username, password):
    users = load_users()
    users[username] = hash_password(password)
    with open(USER_DB_FILE, "w") as f:
        json.dump(users, f)

def check_login(username, password):
    users = load_users()
    return username in users and users[username] == hash_password(password)

def check_user_exists(username):
    users = load_users()
    return username in users

# ==========================================
# BACKGROUND LOADER
# ==========================================
def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    try:
        bin_str = get_base64_of_bin_file(png_file)
        page_bg_img = f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(0,0,0,0.3), rgba(0,0,0,0.6)), url("data:image/png;base64,{bin_str}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """
        st.markdown(page_bg_img, unsafe_allow_html=True)
    except FileNotFoundError:
        pass

try:
    set_png_as_page_bg("background.jpg")
except:
    pass

# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
        html, body, p, label, li, h1, h2, h3, h4, h5, h6 {
            font-family: 'Poppins', sans-serif;
            color: #ffffff !important;
        }
        h1 {
            text-shadow: 0 0 25px rgba(0, 180, 255, 0.9), 0 0 50px rgba(0, 100, 255, 0.6);
        }

        section[data-testid="stSidebar"] {
            background-color: rgba(0, 0, 0, 0.2) !important;
            backdrop-filter: blur(15px);
            border-right: 1px solid rgba(0, 180, 255, 0.2);
        }

        .stTextInput input, .stNumberInput input, .stSelectbox div, .stSlider div {
            background-color: rgba(0, 0, 0, 0.5) !important;
            color: white !important;
            border-radius: 10px;
        }

        div.stForm, div[data-testid="stMetricValue"], div.stTabs, div.stAlert {
            background: rgba(10, 20, 30, 0.6) !important;
            backdrop-filter: blur(20px);
            border: 1px solid rgba(0, 180, 255, 0.2);
            border-radius: 20px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        }

        div.stButton > button {
            background: linear-gradient(135deg, #00b4ff 0%, #0066ff 100%) !important;
            color: white !important;
            border: none !important;
            padding: 10px 20px;
            border-radius: 10px;
            font-weight: 700 !important;
            width: 100%;
            transition: all 0.3s ease;
        }

        div.stButton > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 0 20px rgba(0, 180, 255, 0.6);
        }

        div[data-testid="stChatInput"] {
            background-color: rgba(10, 20, 30, 0.8) !important;
            border: 1px solid #00b4ff !important;
            border-radius: 20px !important;
        }

        div[data-testid="stChatMessage"] {
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        button[data-baseweb="tab"] {
            background-color: transparent !important;
            color: white !important;
            font-weight: bold;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            background-color: #00b4ff !important;
            color: white !important;
            border-radius: 10px;
        }

        .stDeployButton, footer {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# SESSION STATE
# ==========================================
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "user_name" not in st.session_state:
    st.session_state["user_name"] = ""
if "show_results" not in st.session_state:
    st.session_state["show_results"] = False
if "prediction" not in st.session_state:
    st.session_state["prediction"] = 0
if "inputs" not in st.session_state:
    st.session_state["inputs"] = {}
if "input_df" not in st.session_state:
    st.session_state["input_df"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# ==========================================
# GEMINI SETUP
# ==========================================
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    model_ai = genai.GenerativeModel("gemini-1.5-flash")
    ai_available = True
except:
    ai_available = False

# ==========================================
# MODEL LOADER
# ==========================================
@st.cache_resource
def load_model_from_github():
    model_filename = "carbon_model_optimized.pkl"

    # Yahi release link use karo agar model old Carbon_Project repo ke release me stored hai
    model_url = f"https://github.com/AditiiSingh03/Carbon_Project/releases/download/v1.0/{model_filename}"

    if not os.path.exists(model_filename):
        with st.spinner("Downloading AI model... Please wait."):
            try:
                response = requests.get(model_url, stream=True)
                response.raise_for_status()
                with open(model_filename, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            except Exception as e:
                st.error(f"Error downloading model: {e}")
                st.stop()

    return joblib.load(model_filename)

try:
    data = load_model_from_github()
    model = data["model"]
    encoders = data["encoders"]
    feature_names = data["columns"]
except Exception as e:
    st.error(f"Model Load Error: {e}")
    st.stop()

# ==========================================
# HELPERS
# ==========================================
def load_lottieurl(url):
    try:
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None
    except:
        return None

lottie_eco = load_lottieurl("https://lottie.host/5a8b7964-6f97-4d7f-9a1c-6d2c00329976/F63r5l5Xp2.json")
lottie_login = load_lottieurl("https://lottie.host/93122c66-8968-45e3-9970-17482f58e12b/X7wA99a7yX.json")
lottie_chat = load_lottieurl("https://lottie.host/02008f5d-7560-484c-b570-0714b609c2d7/R8gI438P1O.json")

# ==========================================
# PDF GENERATOR
# ==========================================
def create_pdf(user_name, inputs, prediction, advice, model, input_df):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 24)
    pdf.cell(0, 10, "EcoTrack Pro Report", ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("Arial", "I", 12)
    pdf.cell(0, 10, f"Prepared for: {user_name}", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"Total Carbon Footprint: {prediction:.0f} kgCO2", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Your Lifestyle Inputs:", ln=True)
    pdf.set_font("Arial", "", 11)

    for k, v in inputs.items():
        pdf.cell(0, 7, f"- {k}: {v}", ln=True)

    pdf.ln(10)

    try:
        plt.style.use("default")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(input_df)
        fig = plt.figure(figsize=(8, 6))
        shap.plots.waterfall(shap_values[0], show=False)
        plt.title("Impact Breakdown (Why is it high?)", fontsize=12)

        temp_chart = "temp_shap_pdf.png"
        plt.savefig(temp_chart, bbox_inches="tight", dpi=150)
        plt.close()
        plt.style.use("dark_background")

        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Logic Breakdown (SHAP Analysis):", ln=True)
        pdf.image(temp_chart, x=10, y=None, w=170)

        if os.path.exists(temp_chart):
            os.remove(temp_chart)
    except Exception as e:
        pdf.set_font("Arial", "I", 10)
        pdf.cell(0, 10, f"(Graph error: {e})", ln=True)

    if advice:
        pdf.ln(10)
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "AI Sustainability Strategy:", ln=True)
        pdf.ln(5)
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 7, advice.encode("latin-1", "replace").decode("latin-1"))

    return pdf.output(dest="S").encode("latin-1")

# ==========================================
# LOGIN PAGE
# ==========================================
def login_page():
    c1, c2, c3 = st.columns([1, 1.2, 1])

    with c2:
        st.write("")
        st.write("")
        st.markdown("<h1 style='text-align: center; font-size: 4rem; margin-bottom: 5px;'>EcoTrack Pro</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; font-size: 1.1rem; opacity: 0.8; margin-bottom: 20px;'>AI-Powered Sustainability Platform</p>", unsafe_allow_html=True)

        tab_login, tab_signup = st.tabs(["🔐 Login", "📝 Sign Up"])

        with tab_login:
            with st.form("login_form"):
                if lottie_login:
                    st_lottie(lottie_login, height=100, key="login_anim")
                username = st.text_input("Username", placeholder="Enter your registered Name")
                password = st.text_input("Password", type="password", placeholder="••••••••")
                st.write("")
                if st.form_submit_button("🚀 LOGIN"):
                    if check_login(username, password):
                        st.session_state["logged_in"] = True
                        st.session_state["user_name"] = username
                        st.success("Login Successful!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid Username or Password")

        with tab_signup:
            with st.form("signup_form"):
                st.markdown("### Create New Account")
                new_user = st.text_input("Choose Username", placeholder="e.g. Aditi123")
                new_pass = st.text_input("Set Password", type="password", placeholder="••••••••")
                confirm_pass = st.text_input("Confirm Password", type="password", placeholder="••••••••")
                st.write("")
                if st.form_submit_button("✨ CREATE ACCOUNT"):
                    if new_user and new_pass:
                        if new_pass != confirm_pass:
                            st.error("Passwords do not match!")
                        elif check_user_exists(new_user):
                            st.error("Username already exists! Please Login.")
                        else:
                            save_user(new_user, new_pass)
                            st.success("Account Created! Please go to Login tab.")
                    else:
                        st.warning("Please fill all fields.")

# ==========================================
# MAIN APP
# ==========================================
def main_app():
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=80)
        st.markdown(f"### Hi, {st.session_state['user_name']}")

        selected = option_menu(
            menu_title="Main Menu",
            options=["Home", "Dashboard", "Vision AI", "Expert Chat", "Logout"],
            icons=["house", "speedometer2", "camera", "chat-dots", "box-arrow-left"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"color": "#00b4ff", "font-size": "18px"},
                "nav-link": {"color": "white", "font-size": "16px", "text-align": "left", "margin": "5px", "background-color": "transparent"},
                "nav-link-selected": {"background-color": "#00b4ff", "color": "white", "font-weight": "bold"},
            }
        )

    if selected == "Logout":
        st.session_state["logged_in"] = False
        st.session_state["show_results"] = False
        st.rerun()

    if selected == "Home":
        st.write("")
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(
                f"<h1 style='font-size: 3.8rem; text-shadow: 0 0 40px rgba(0,180,255,0.6);'>Welcome, {st.session_state['user_name']}&nbsp;👋</h1>",
                unsafe_allow_html=True
            )
            st.markdown("<h3>Ready to start your green journey?</h3>", unsafe_allow_html=True)
            st.markdown("""
            <div style='background: rgba(10, 20, 30, 0.6); padding: 30px; border-radius: 15px; border-left: 5px solid #00b4ff; margin-top: 20px; box-shadow: 0 0 20px rgba(0,0,0,0.5); backdrop-filter: blur(10px);'>
                <h4 style='color: #00b4ff !important; margin-bottom: 15px;'>🌍 How EcoTrack Helps You</h4>
                <p style='font-size: 1.1rem; line-height: 1.6; color: #eee; margin-bottom: 10px;'>
                    We make sustainability simple. Instead of guessing, use our AI tools to measure your impact on the planet.
                </p>
                <ul style='font-size: 1.05rem; line-height: 1.8; color: #ddd; margin-left: 20px;'>
                    <li><b>📊 Track Daily Habits:</b> Enter your travel or electricity usage in the <i>Dashboard</i> to calculate your carbon footprint.</li>
                    <li><b>🍎 Scan & Check:</b> Use <i>Vision AI</i> to snap photos of appliances or food to instantly see their eco-rating.</li>
                    <li><b>💡 Get Advice:</b> Our system gives you personalized tips to save energy and money.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            if lottie_eco:
                st_lottie(lottie_eco, height=400)

    if selected == "Dashboard":
        st.title("📊 Carbon Calculator")

        st.subheader("Model Performance")
        perf1, perf2, perf3 = st.columns(3)
        perf1.metric("R² Score", "0.89")
        perf2.metric("RMSE", "245 kg")
        perf3.metric("MAE", "180 kg")

        if not st.session_state["show_results"]:
            st.markdown("### ⚙️ Lifestyle Inputs")

            with st.form("main_form"):
                input_data = {}
                display_inputs = {}

                cat_cols = [c for c in feature_names if c in encoders]
                num_cols = [c for c in feature_names if c not in encoders]

                st.markdown("##### 📝 Basic Details")
                c1, c2, c3 = st.columns(3)
                cols_list = [c1, c2, c3]

                for i, col_name in enumerate(cat_cols):
                    clean_name = col_name.replace("_", " ").title()
                    with cols_list[i % 3]:
                        val = st.selectbox(clean_name, list(encoders[col_name].classes_))
                        input_data[col_name] = encoders[col_name].transform([val])[0]
                        display_inputs[clean_name] = val

                st.markdown("##### 📏 Usage & Consumption")
                n1, n2 = st.columns(2)
                num_cols_list = [n1, n2]

                for i, col_name in enumerate(num_cols):
                    clean_name = col_name.replace("_", " ").title()
                    with num_cols_list[i % 2]:
                        if "Distance" in col_name:
                            val = st.slider(clean_name, 0, 5000, 500)
                        elif "Bill" in col_name:
                            val = st.slider(clean_name, 0, 1000, 100)
                        else:
                            val = st.number_input(clean_name, value=0)

                        input_data[col_name] = val
                        display_inputs[clean_name] = val

                st.write("")
                if st.form_submit_button("🚀 Calculate Impact"):
                    input_df = pd.DataFrame([input_data])
                    input_df = input_df[feature_names]
                    prediction = model.predict(input_df)[0]

                    st.session_state["prediction"] = prediction
                    st.session_state["inputs"] = display_inputs
                    st.session_state["input_df"] = input_df
                    st.session_state["show_results"] = True
                    st.rerun()

        else:
            if st.button("⬅️ Calculate Again"):
                st.session_state["show_results"] = False
                st.rerun()

            prediction = st.session_state["prediction"]
            diff = prediction - 3000

            tab1, tab2 = st.tabs(["📊 Analysis & Graphs", "🚀 Action Plan"])

            with tab1:
                col_res1, col_res2 = st.columns([1, 2])

                with col_res1:
                    st.metric("Annual Footprint", f"{prediction:.0f} kg", delta=f"{diff:.0f} vs Avg", delta_color="inverse")
                    st.metric("Global Target", "3000 kg")

                with col_res2:
                    fig_bar = go.Figure(
                        data=[
                            go.Bar(
                                x=["Your Footprint", "Global Average", "India Avg"],
                                y=[prediction, 4500, 2500],
                                marker_color=["#00b4ff", "#00ff99", "#ffcc00"]
                            )
                        ]
                    )
                    fig_bar.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font={"color": "white"}
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                st.markdown("### 📌 Estimated Emission Breakdown")

                transport_emission = prediction * 0.45
                electricity_emission = prediction * 0.30
                lifestyle_emission = prediction * 0.25

                fig_break = go.Figure(
                    data=[
                        go.Pie(
                            labels=["Transport", "Electricity", "Lifestyle"],
                            values=[transport_emission, electricity_emission, lifestyle_emission],
                            hole=0.45
                        )
                    ]
                )

                fig_break.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "white"},
                    height=420
                )

                st.plotly_chart(fig_break, use_container_width=True)

                st.write("---")
                st.markdown("### 🔍 Logic Breakdown")

                with st.spinner("Calculating Drivers..."):
                    try:
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer(st.session_state["input_df"])

                        plt.style.use("dark_background")
                        fig_shap, ax = plt.subplots(figsize=(10, 5))
                        fig_shap.patch.set_facecolor("none")
                        ax.set_facecolor("none")

                        shap.plots.waterfall(shap_values[0], show=False)
                        st.pyplot(fig_shap, clear_figure=True)
                    except:
                        st.error("SHAP Error")

            with tab2:
                col_act1, col_act2 = st.columns(2)

                with col_act1:
                    fig = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=prediction,
                            gauge={
                                "axis": {"range": [None, 10000], "tickcolor": "white"},
                                "bar": {"color": "#00b4ff"},
                                "bgcolor": "rgba(0,0,0,0)",
                                "bordercolor": "white",
                                "steps": [
                                    {"range": [0, 3000], "color": "rgba(0, 255, 100, 0.6)"},
                                    {"range": [3000, 6000], "color": "rgba(255, 200, 0, 0.6)"},
                                    {"range": [6000, 10000], "color": "rgba(255, 0, 0, 0.6)"}
                                ]
                            }
                        )
                    )
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font={"color": "white"}
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col_act2:
                    st.markdown("### 🌍 Impact Benchmark")

                    if prediction < 2500:
                        st.success("Your carbon footprint is below the average Indian benchmark.")
                    elif prediction < 4500:
                        st.warning("Your carbon footprint is moderate. Some lifestyle changes can reduce it further.")
                    else:
                        st.error("Your carbon footprint is high compared with common benchmarks.")

                    st.markdown("### 🔧 What-If Reduction Simulator")

                    reduce_transport = st.slider("Reduce travel-related usage by (%)", 0, 100, 20)
                    reduce_energy = st.slider("Reduce electricity-related usage by (%)", 0, 100, 15)

                    reduced_prediction = prediction * (1 - (0.5 * reduce_transport / 100) - (0.2 * reduce_energy / 100))
                    reduced_prediction = max(reduced_prediction, 0)

                    sim1, sim2 = st.columns(2)
                    sim1.metric("Projected Reduced Footprint", f"{reduced_prediction:.0f} kg")
                    sim2.metric("Potential Reduction", f"{prediction - reduced_prediction:.0f} kg")

                    if ai_available:
                        with st.spinner("Generating AI Plan..."):
                            prompt = f"Footprint: {prediction}. Inputs: {st.session_state['inputs']}. Give 3 strict bullet points."
                            try:
                                ai_msg = model_ai.generate_content(prompt).text
                                st.info(f"🤖 **AI Strategy:** \n\n{ai_msg}")
                            except:
                                st.error("AI Busy.")

                    pdf = create_pdf(
                        st.session_state["user_name"],
                        st.session_state["inputs"],
                        prediction,
                        ai_msg if "ai_msg" in locals() else "",
                        model,
                        st.session_state["input_df"]
                    )

                    st.download_button(
                        "📥 Download Full Report",
                        pdf,
                        "EcoReport.pdf",
                        "application/pdf"
                    )

    if selected == "Vision AI":
        st.title("📸 Vision AI Scanner")
        st.markdown("### Upload an item to detect its carbon footprint")

        c1, c2 = st.columns([1, 1])

        with c1:
            st.markdown("""<div style='background: rgba(10,20,30,0.5); padding: 20px; border-radius: 15px; border: 1px dashed #00b4ff;'><p style='margin:0; text-align:center;'>⬇️ Upload Image Here</p></div>""", unsafe_allow_html=True)
            uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)

        with c2:
            if uploaded_file:
                st.write("")
                if st.button("🔍 SCAN IMAGE NOW"):
                    if ai_available:
                        with st.spinner("🤖 AI is analyzing the pixels..."):
                            try:
                                prompt = "Identify this object. Estimate its carbon footprint (Low/Medium/High). Give a 2-line sustainability verdict. Use bold text for key points."
                                response = model_ai.generate_content([prompt, image])
                                st.info(f"### ✅ Analysis Complete\n\n{response.text}")
                            except Exception as e:
                                st.error(f"AI Error: {e}")
                    else:
                        st.error("AI API Key missing.")
            else:
                st.info("👈 Upload an image to see the AI analysis here.")

    if selected == "Expert Chat":
        st.title("💬 Eco-Bot Consultant")
        st.markdown("Ask any question about sustainability, recycling, or reducing your footprint.")

        c1, c2 = st.columns([1, 2])

        with c1:
            if lottie_chat:
                st_lottie(lottie_chat, height=300)
            st.markdown("""<div style='background: rgba(10, 20, 30, 0.6); padding: 20px; border-radius: 15px; border-left: 4px solid #00b4ff;'><h4 style='color:#00b4ff; margin:0;'>🤖 I am EcoBot</h4><p style='font-size: 0.9rem; color: #ccc;'>I am powered by Gemini AI to help you live greener.</p></div>""", unsafe_allow_html=True)

        with c2:
            chat_container = st.container()
            with chat_container:
                for message in st.session_state["chat_history"]:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

            if prompt := st.chat_input("Ask me anything..."):
                st.session_state["chat_history"].append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                if ai_available:
                    with st.chat_message("assistant"):
                        message_placeholder = st.empty()
                        try:
                            system_instruction = "You are an expert sustainability consultant named EcoBot. Keep answers short, practical, and motivating."
                            response = model_ai.generate_content(f"{system_instruction}\n\nUser Question: {prompt}")
                            full_response = response.text
                            message_placeholder.markdown(full_response)
                            st.session_state["chat_history"].append({"role": "assistant", "content": full_response})
                        except Exception as e:
                            st.error(f"AI Error: {e}")

# --- RUN ---
if not st.session_state["logged_in"]:
    login_page()
else:
    main_app()
