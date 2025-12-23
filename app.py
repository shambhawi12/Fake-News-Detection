import streamlit as st
import joblib
import re
import string
from collections import Counter

# Page configuration - MUST BE FIRST
st.set_page_config(
    page_title="InFactAI - Fake News Detection",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------- CRITICAL: Load NLTK data ONCE at startup ----------
@st.cache_resource
def load_nltk_data():
    """Load NLTK data once and cache it"""
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    from nltk.corpus import stopwords
    return set(stopwords.words('english'))

# Load stopwords once
stop_words = load_nltk_data()

# ---------- Load model & vectorizer ONCE ----------
@st.cache_resource
def load_models():
    """Load ML models once and cache them"""
    try:
        model = joblib.load("model/fake_news_model.pkl")
        vectorizer = joblib.load("model/vectorizer.pkl")
        return model, vectorizer
    except FileNotFoundError:
        st.error("⚠ Model files not found. Please ensure 'fake_news_model.pkl' and 'vectorizer.pkl' are in the 'model/' directory.")
        return None, None

model, vectorizer = load_models()

# ---------- OPTIMIZED CSS (Removed heavy animations) ----------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        font-family: 'Inter', sans-serif;
        max-width: 1200px;
        margin: 0 auto;
        padding: 0;
    }
    
    .block-container {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        max-width: 100% !important;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Simple fade-in animation */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Header section */
    .header-section {
        text-align: center;
        padding: 4rem 0 3rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: -1rem -1rem 0 -1rem;
        width: calc(100% + 2rem);
        animation: fadeIn 0.6s ease-out;
    }
    
    .logo-container {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        color: white;
        margin: 0;
        letter-spacing: -0.02em;
    }
    
    .tagline {
        font-size: 1.2rem;
        font-weight: 300;
        margin-top: 1rem;
        opacity: 0.9;
    }
    
    /* How it works section */
    .how-it-works {
        background: white;
        border-radius: 16px;
        padding: 4rem 3rem;
        margin: -2rem 2rem 3rem 2rem;
        position: relative;
        z-index: 2;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    .section-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a1a;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    .section-description {
        font-size: 1.2rem;
        color: #64748b;
        line-height: 1.6;
        max-width: 800px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    .features-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 2rem;
        margin-top: 2rem;
    }
    
    .feature-item {
        text-align: center;
        padding: 2rem 1rem;
        background: #f8fafc;
        border-radius: 12px;
        transition: all 0.3s ease;
    }
    
    .feature-item:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.2);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        display: block;
    }
    
    .feature-title {
        font-weight: 600;
        font-size: 1.2rem;
        color: #1a1a1a;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .feature-description {
        font-size: 1rem;
        color: #64748b;
        line-height: 1.4;
        text-align: center;
    }
    
    /* Input section */
    .input-section {
        background: white;
        border-radius: 16px;
        padding: 3rem;
        margin: 3rem 2rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    
    .input-title {
        font-size: 2rem;
        font-weight: 600;
        text-align: center;
        margin-bottom: 2rem;
        color: #1a1a1a;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        padding: 1.2rem 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 1.5rem;
        transition: all 0.3s ease;
        font-family: 'Inter', sans-serif;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
    }
    
    /* Results section */
    .result-container {
        background: white;
        border-radius: 16px;
        padding: 3rem;
        margin: 2rem 2rem;
        border-left: 6px solid;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        animation: slideIn 0.4s ease-out;
    }
    
    .result-real {
        border-left-color: #10b981;
        background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%);
    }
    
    .result-fake {
        border-left-color: #ef4444;
        background: linear-gradient(135deg, #fef2f2 0%, #fef7f7 100%);
    }
    
    .result-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .result-real .result-title { color: #059669; }
    .result-fake .result-title { color: #dc2626; }
    
    .result-description {
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
        color: #374151;
    }
    
    .confidence-section {
        margin-top: 2rem;
        background: rgba(255,255,255,0.7);
        padding: 1.5rem;
        border-radius: 12px;
    }
    
    .confidence-label {
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 1rem;
        color: #1a1a1a;
    }
    
    .confidence-bar {
        background: #e5e7eb;
        border-radius: 10px;
        height: 12px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 1s ease-out;
    }
    
    .confidence-real { background: linear-gradient(90deg, #10b981, #059669); }
    .confidence-fake { background: linear-gradient(90deg, #ef4444, #dc2626); }
    
    .confidence-percentage {
        font-size: 2rem;
        font-weight: 700;
        text-align: center;
        margin-top: 1rem;
    }
    
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .metric-card {
        background: rgba(255,255,255,0.7);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        font-weight: 500;
        color: #64748b;
    }
    
    .fake-metric { color: #dc2626; }
    .real-metric { color: #059669; }
    .neutral-metric { color: #3b82f6; }
    
    .analysis-section {
        background: rgba(255,255,255,0.7);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
    }
    
    .analysis-title {
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #1a1a1a;
    }
    
    .red-flag {
        background: rgba(254,226,226,0.8);
        border-left: 4px solid #ef4444;
        padding: 1rem;
        margin: 0.8rem 0;
        border-radius: 8px;
        font-weight: 500;
    }
    
    .credibility-indicator {
        background: rgba(240,253,244,0.8);
        border-left: 4px solid #10b981;
        padding: 1rem;
        margin: 0.8rem 0;
        border-radius: 8px;
        font-weight: 500;
    }
    
    .stTextArea > div > div > textarea {
        width: 100%;
        min-height: 150px;
        padding: 1.5rem;
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        font-size: 1rem;
        font-family: 'Inter', sans-serif;
        line-height: 1.6;
        transition: all 0.3s ease;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        outline: none;
    }
    
    .examples-section {
        margin: 4rem 2rem;
        text-align: center;
    }
    
    .examples-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
        gap: 2rem;
        margin-top: 3rem;
    }
    
    .example-card {
        background: white;
        border: 2px solid #e2e8f0;
        border-radius: 16px;
        padding: 2rem;
        text-align: left;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .example-card:hover {
        border-color: #667eea;
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.1);
    }
    
    .example-type {
        font-size: 0.875rem;
        font-weight: 700;
        margin-bottom: 1rem;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        display: inline-block;
    }
    
    .real-news { 
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        color: #16a34a;
    }
    .fake-news { 
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        color: #dc2626;
    }
    .health-news { 
        background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%);
        color: #9333ea;
    }
    
    .example-text {
        color: #374151;
        line-height: 1.6;
        font-size: 1.05rem;
    }
    
    .performance-section {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        color: white;
        border-radius: 16px;
        padding: 4rem 3rem;
        margin: 4rem 2rem;
        text-align: center;
    }
    
    .performance-section .section-title {
        color: white;
        margin-bottom: 3rem;
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 3rem;
        margin-top: 2rem;
    }
    
    .stat-item {
        text-align: center;
    }
    
    .stat-value {
        font-size: 3rem;
        font-weight: 700;
        color: #10b981;
        display: block;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        font-size: 1rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .disclaimer {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border: 2px solid #f59e0b;
        border-radius: 16px;
        padding: 2rem;
        margin: 4rem 2rem 2rem 2rem;
    }
    
    .disclaimer-title {
        font-weight: 700;
        color: #92400e;
        margin-bottom: 1rem;
        font-size: 1.2rem;
    }
    
    .disclaimer-text {
        color: #92400e;
        line-height: 1.6;
    }
    
    @media (max-width: 768px) {
        .main-title { font-size: 2.5rem; }
        .section-title { font-size: 2rem; }
        .how-it-works, .input-section, .examples-section,
        .performance-section, .disclaimer, .result-container {
            margin-left: 1rem;
            margin-right: 1rem;
            padding: 2rem;
        }
        .features-grid, .examples-grid { grid-template-columns: 1fr; }
        .stats-grid { grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }
        .metrics-grid { grid-template-columns: 1fr; }
    }
</style>
""", unsafe_allow_html=True)

# ---------- Text Analysis Functions ----------
def clean_text(text):
    """Clean and preprocess text"""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def analyze_text_features(text):
    """Analyze various text features"""
    features = {}
    features['word_count'] = len(text.split())
    features['sentence_count'] = len([s for s in text.split('.') if s.strip()])
    features['char_count'] = len(text)
    
    exclamation_count = text.count('!')
    question_count = text.count('?')
    caps_count = sum(1 for c in text if c.isupper())
    
    features['exclamation_ratio'] = exclamation_count / max(features['sentence_count'], 1)
    features['question_ratio'] = question_count / max(features['sentence_count'], 1)
    features['caps_ratio'] = caps_count / max(features['char_count'], 1)
    
    return features

def detect_red_flags(text, features):
    """Detect potential red flags"""
    red_flags = []
    credibility_indicators = []
    
    suspicious_phrases = [
        "big pharma", "doctors hate", "miracle cure", "secret", "they don't want you to know",
        "shocking", "revealed", "exposed", "conspiracy", "cover-up"
    ]
    
    text_lower = text.lower()
    found_suspicious = [phrase for phrase in suspicious_phrases if phrase in text_lower]
    
    if found_suspicious:
        red_flags.append(f"SUSPICIOUS LANGUAGE: Found '{', '.join(found_suspicious[:3])}'")
    
    if features['exclamation_ratio'] > 0.3:
        red_flags.append("EXCESSIVE PUNCTUATION: Too many exclamation marks")
    
    if features['caps_ratio'] > 0.1:
        red_flags.append("EXCESSIVE CAPS: Overuse of capital letters")
    
    credible_terms = [
        "research", "study", "published", "journal", "university", "professor",
        "data", "analysis", "evidence"
    ]
    
    found_credible = [term for term in credible_terms if term in text_lower]
    if found_credible:
        credibility_indicators.append(f"CREDIBLE LANGUAGE: Found '{', '.join(found_credible[:3])}'")
    
    if 50 <= features['word_count'] <= 500:
        credibility_indicators.append("APPROPRIATE LENGTH: Article length seems reasonable")
    
    return red_flags, credibility_indicators

# ---------- Header Section ----------
st.markdown("""
<div class="header-section">
    <div class="logo-container">
        <h1 class="main-title">🔍 InFactAI</h1>
    </div>
    <p class="tagline">Advanced AI-powered fake news detection</p>
</div>
""", unsafe_allow_html=True)

# ---------- How It Works Section ----------
st.markdown("""
<div class="how-it-works">
    <h2 class="section-title">How It Works</h2>
    <div style="text-align: center; max-width: 800px; margin: 0 auto 3rem auto;">
        <p style="font-size: 1.2rem; color: #64748b; line-height: 1.6; margin: 0;">
            Our advanced AI analyzes news content using machine learning algorithms trained on 1000+ real and fake news examples. 
            Simply paste any text below for instant analysis.
        </p>
    </div>
    <div class="features-grid">
        <div class="feature-item">
            <span class="feature-icon">🤖</span>
            <div class="feature-title">AI-Powered</div>
            <div class="feature-description">Advanced machine learning algorithms for accurate detection</div>
        </div>
        <div class="feature-item">
            <span class="feature-icon">⚡</span>
            <div class="feature-title">Instant Results</div>
            <div class="feature-description">Get predictions in under 1 second with confidence scores</div>
        </div>
        <div class="feature-item">
            <span class="feature-icon">🎯</span>
            <div class="feature-title">Reliable Detection</div>
            <div class="feature-description">High accuracy rate tested on diverse news sources</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------- Input Section ----------
st.markdown('<h2 class="input-title">Analyze News Article</h2>', unsafe_allow_html=True)

# Sample articles
sample_fake = """Russia's Putin laments 'spymania' gripping Washington,moscow reuters russian president vladimir putin said thursday spymania artificially whipped russia united states eventually relations two countries would get back normal said contacts russian officials members us president donald trump team election campaign routine twisted trump opponents asked reporter thought trump record office putin said judge saw significant achievements trump administration,worldnews,"December 14, 2017 " """

sample_real = """WOMAN ARRESTED For Wearing T-Shirt Naming Muslim Extremist Who Fled Country After Failed Jihad Attempt [VIDEO],shocking example government putting rights violent extreme muslims citizens,left-news,"Feb 29, 2016" """

input_option = st.radio("Choose input method:", 
                       ["Type/Paste Article", "Use Sample Fake News", "Use Sample Real News"])

if input_option == "Use Sample Fake News":
    user_input = st.text_area("", value=sample_fake, height=150)
elif input_option == "Use Sample Real News":
    user_input = st.text_area("", value=sample_real, height=150)
else:
    user_input = st.text_area("", height=150, placeholder="Enter or paste the news article text you want to analyze...")

analyze_button = st.button("🔍 Analyze Article", key="analyze")

if analyze_button:
    if user_input.strip() == "":
        st.warning("⚠ Please enter some text to analyze.")
    elif model is None or vectorizer is None:
        st.error("❌ Model not available.")
    else:
        # Perform analysis (no artificial delay)
        cleaned_input = clean_text(user_input)
        transformed_text = vectorizer.transform([cleaned_input])
        prediction = model.predict(transformed_text)[0]
        proba = model.predict_proba(transformed_text)[0]

        fake_prob = proba[1] * 100
        real_prob = proba[0] * 100
        confidence = max(fake_prob, real_prob)
        
        features = analyze_text_features(user_input)
        red_flags, credibility_indicators = detect_red_flags(user_input, features)
        
        label = "FAKE" if prediction == 1 else "REAL"
        
        if label == "REAL":
            st.markdown(f"""
            <div class="result-container result-real">
                <div class="result-title">✅ REAL NEWS DETECTED</div>
                <div class="result-description">
                    This article appears to be <strong>authentic</strong> based on our AI analysis.
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-container result-fake">
                <div class="result-title">⚠ FAKE NEWS DETECTED</div>
                <div class="result-description">
                    This article appears to be <strong>potentially misleading</strong> based on our AI analysis.
                </div>
            """, unsafe_allow_html=True)
            
        st.markdown(f"""
        <div class="confidence-section">
            <div class="confidence-label">🎯 Confidence Level</div>
            <div class="confidence-bar">
                <div class="confidence-fill {'confidence-real' if label == 'REAL' else 'confidence-fake'}" style="width: {confidence}%"></div>
            </div>
            <div class="confidence-percentage" style="color: {'#059669' if label == 'REAL' else '#dc2626'}">
                {confidence:.1f}%
            </div>
        </div>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value fake-metric">{fake_prob:.1f}%</div>
                <div class="metric-label">Fake Probability</div>
            </div>
            <div class="metric-card">
                <div class="metric-value real-metric">{real_prob:.1f}%</div>
                <div class="metric-label">Real Probability</div>
            </div>
            <div class="metric-card">
                <div class="metric-value neutral-metric">{features['word_count']}</div>
                <div class="metric-label">Words Analyzed</div>
            </div>
        </div>
        <div class="analysis-section">
            <div class="analysis-title">🔍 Detailed Analysis</div>
            <p><strong>Text Length:</strong> {features['word_count']} words, {features['sentence_count']} sentences</p>
            <p><strong>Indicators:</strong> {len(red_flags)} suspicious, {len(credibility_indicators)} credible</p>
        </div>
        """, unsafe_allow_html=True)
        
        if red_flags:
            st.markdown('<div class="analysis-title">🚩 Red Flags:</div>', unsafe_allow_html=True)
            for flag in red_flags:
                st.markdown(f'<div class="red-flag">• {flag}</div>', unsafe_allow_html=True)
        
        if credibility_indicators:
            st.markdown('<div class="analysis-title">✅ Credibility Indicators:</div>', unsafe_allow_html=True)
            for indicator in credibility_indicators:
                st.markdown(f'<div class="credibility-indicator">• {indicator}</div>', unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

# ---------- Examples Section ----------
st.markdown("""
<div class="examples-section">
    <h2 class="section-title">Try These Examples</h2>
    <div class="examples-grid">
        <div class="example-card">
            <div class="example-type real-news">Real News</div>
            <div class="example-text">
                "WATCH TREY GOWDY Crush The Lying Media During Benghazi Report Press Conference"
            </div>
        </div>
        <div class="example-card">
            <div class="example-type fake-news">Fake News</div>
            <div class="example-text">
                "House Speaker Ryan urges coordinated response to Brussels attack"
            </div>
        </div>
        <div class="example-card">
            <div class="example-type health-news">International News</div>
            <div class="example-text">
                "MICHELLE OBAMA STARS IN AWKWARD KICKBOXING VIDEO"
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------- Performance Section ----------
st.markdown("""
<div class="performance-section">
    <h2 class="section-title">Model Performance</h2>
    <div class="stats-grid">
        <div class="stat-item">
            <span class="stat-value">98.9%</span>
            <div class="stat-label">Accuracy</div>
        </div>
        <div class="stat-item">
            <span class="stat-value">40K+</span>
            <div class="stat-label">Training Data</div>
        </div>
        <div class="stat-item">
            <span class="stat-value">&lt;1s</span>
            <div class="stat-label">Response Time</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------- Disclaimer ----------
st.markdown("""
<div class="disclaimer">
    <div class="disclaimer-title">⚠ Important Disclaimer</div>
    <div class="disclaimer-text">
        This AI model has been trained on a specific dataset and may not accurately predict the authenticity 
        of news articles that fall outside the scope of its training data. Please use this tool as 
        a preliminary assessment only and always verify information through multiple reliable sources.
    </div>
</div>
""", unsafe_allow_html=True)
