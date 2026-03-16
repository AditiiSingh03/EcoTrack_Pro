import google.generativeai as genai

# Aapki API Key
GOOGLE_API_KEY = "AIzaSyDOSfW102xXeYYVLsaGVdmhKTMoO1mbgIQ"
genai.configure(api_key=GOOGLE_API_KEY)

print("🔍 Searching for available models...")

try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"✅ Available: {m.name}")
except Exception as e:
    print(f"❌ Error: {e}")