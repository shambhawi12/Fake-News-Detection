import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
import time
from collections import Counter

# Page configuration
st.set_page_config(
    page_title="InFactAI - Fake News Detection",
    page_icon= None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS to match the reference design exactly with enhanced animations
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styling */
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
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Keyframe Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% {
            transform: translateY(0);
        }
        40% {
            transform: translateY(-10px);
        }
        60% {
            transform: translateY(-5px);
        }
    }
    
    @keyframes pulse {
        0% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
        100% {
            transform: scale(1);
        }
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    @keyframes glow {
        0%, 100% {
            box-shadow: 0 0 5px rgba(102, 126, 234, 0.5);
        }
        50% {
            box-shadow: 0 0 20px rgba(102, 126, 234, 0.8), 0 0 30px rgba(102, 126, 234, 0.6);
        }
    }
    
    @keyframes float {
        0%, 100% {
            transform: translateY(0px);
        }
        50% {
            transform: translateY(-10px);
        }
    }
    
    @keyframes shimmer {
        0% {
            background-position: -200% 0;
        }
        100% {
            background-position: 200% 0;
        }
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
    
    /* Animated Cube Logo */
    .logo-container {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
        animation: fadeInUp 1s ease-out 0.3s both;
    }
    
    .animated-cube {
        width: 50px;
        height: 50px;
        position: relative;
        transform-style: preserve-3d;
        animation: rotateCube 4s linear infinite;
    }
    
    @keyframes rotateCube {
        0% { transform: rotateX(0deg) rotateY(0deg); }
        100% { transform: rotateX(360deg) rotateY(360deg); }
    }
    
    .cube-face {
        position: absolute;
        width: 50px;
        height: 50px;
        background: rgba(255, 255, 255, 0.1);
        border: 2px solid rgba(255, 255, 255, 0.3);
        backdrop-filter: blur(5px);
    }
    
    .cube-face.front  { transform: translateZ(25px); }
    .cube-face.back   { transform: rotateY(180deg) translateZ(25px); }
    .cube-face.right  { transform: rotateY(90deg) translateZ(25px); }
    .cube-face.left   { transform: rotateY(-90deg) translateZ(25px); }
    .cube-face.top    { transform: rotateX(90deg) translateZ(25px); }
    .cube-face.bottom { transform: rotateX(-90deg) translateZ(25px); }
    
    /* Header section with gradient and animations */
    .header-section {
        text-align: center;
        padding: 4rem 0 3rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: -1rem -1rem 0 -1rem;
        width: calc(100% + 2rem);
        animation: fadeIn 1s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .header-section::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        animation: shimmer 6s linear infinite;
        pointer-events: none;
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        color: white;
        margin: 0;
        letter-spacing: -0.02em;
    }
    
    .main-logo {
        font-size: 4rem;
        margin-bottom: 1rem;
        display: block;
        animation: bounce 2s infinite;
    }
    
    .tagline {
        font-size: 1.2rem;
        font-weight: 300;
        margin-top: 1rem;
        opacity: 0.9;
        animation: fadeInUp 1s ease-out 0.6s both;
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
        animation: fadeInUp 1s ease-out 0.8s both;
    }
    
    .section-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a1a;
        margin-bottom: 1.5rem;
        animation: fadeInUp 0.8s ease-out;
        text-align: center;
    }
    
    .section-description {
        font-size: 1.2rem;
        color: #64748b;
        line-height: 1.6;
        max-width: 800px;
        margin: 0 auto 3rem auto;
        animation: fadeInUp 0.8s ease-out 0.2s both;
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
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeInUp 0.8s ease-out both;
        position: relative;
        overflow: hidden;
    }
    
    .feature-item:nth-child(1) { animation-delay: 0.4s; }
    .feature-item:nth-child(2) { animation-delay: 0.6s; }
    .feature-item:nth-child(3) { animation-delay: 0.8s; }
    
    .feature-item::before {
        content: "";
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.1), transparent);
        transition: left 0.5s ease;
    }
    
    .feature-item:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.2);
        background: linear-gradient(135deg, #f8fafc 0%, #f0f4ff 100%);
    }
    
    .feature-item:hover::before {
        left: 100%;
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        display: block;
        /* Removed bounce animation */
    }
    
    .feature-title {
        font-weight: 600;
        font-size: 1.2rem;
        color: #1a1a1a;
        margin-bottom: 0.5rem;
        transition: color 0.3s ease;
        text-align: center;
    }
    
    .feature-item:hover .feature-title {
        color: #667eea;
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
        animation: slideInLeft 1s ease-out;
        transition: all 0.3s ease;
    }
    
    .input-section:hover {
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        transform: translateY(-2px);
    }
    
    .input-title {
        font-size: 2rem;
        font-weight: 600;
        text-align: center;
        margin-bottom: 2rem;
        color: #1a1a1a;
        animation: fadeInUp 0.8s ease-out;
    }
    
    /* Radio button styling */
    .stRadio > div {
        display: flex;
        gap: 1rem;
        justify-content: center;
        margin-bottom: 1.5rem;
    }
    
    .stRadio > div > label {
        background: #f8fafc;
        padding: 0.8rem 1.5rem;
        border-radius: 25px;
        border: 2px solid #e2e8f0;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        font-weight: 500;
        position: relative;
        overflow: hidden;
    }
    
    .stRadio > div > label::before {
        content: "";
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.1), transparent);
        transition: left 0.5s ease;
    }
    
    .stRadio > div > label:hover {
        border-color: #667eea;
        background: #f0f4ff;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
    }
    
    .stRadio > div > label:hover::before {
        left: 100%;
    }
    
    /* Button styling with enhanced animations */
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
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        font-family: 'Inter', sans-serif;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: "";
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #7c3aed 0%, #8b5cf6 100%);
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:active {
        transform: translateY(-1px) scale(1.01);
    }
    
    /* Results section with enhanced animations */
    .result-container {
        background: white;
        border-radius: 16px;
        padding: 3rem;
        margin: 2rem 2rem;
        border-left: 6px solid;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        animation: slideIn 0.6s cubic-bezier(0.4, 0, 0.2, 1), pulse 0.3s ease-out 0.6s;
        position: relative;
        overflow: hidden;
    }
    
    .result-container::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        animation: shimmer 3s linear infinite;
        pointer-events: none;
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
        animation: fadeInUp 0.6s ease-out 0.3s both;
    }
    
    .result-real .result-title { 
        color: #059669;
        text-shadow: 0 2px 4px rgba(16, 185, 129, 0.1);
    }
    .result-fake .result-title { 
        color: #dc2626;
        text-shadow: 0 2px 4px rgba(239, 68, 68, 0.1);
    }
    
    .result-description {
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
        color: #374151;
        animation: fadeInUp 0.6s ease-out 0.5s both;
    }
    
    .confidence-section {
        margin-top: 2rem;
        background: rgba(255,255,255,0.7);
        padding: 1.5rem;
        border-radius: 12px;
        animation: fadeInUp 0.6s ease-out 0.7s both;
        backdrop-filter: blur(5px);
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
        position: relative;
        animation: fadeIn 0.3s ease-out 1s both;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 2s cubic-bezier(0.4, 0, 0.2, 1) 0.5s;
        position: relative;
        animation: shimmer 2s linear infinite;
    }
    
    .confidence-real { 
        background: linear-gradient(90deg, #10b981, #059669, #10b981);
        background-size: 200% 100%;
    }
    .confidence-fake { 
        background: linear-gradient(90deg, #ef4444, #dc2626, #ef4444);
        background-size: 200% 100%;
    }
    
    .confidence-percentage {
        font-size: 2rem;
        font-weight: 700;
        text-align: center;
        margin-top: 1rem;
        animation: fadeInUp 0.6s ease-out 1.2s both, pulse 0.5s ease-out 1.8s;
    }
    
    /* Metrics grid with staggered animations */
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
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeInUp 0.6s ease-out both;
        backdrop-filter: blur(5px);
    }
    
    .metric-card:nth-child(1) { animation-delay: 0.9s; }
    .metric-card:nth-child(2) { animation-delay: 1.1s; }
    .metric-card:nth-child(3) { animation-delay: 1.3s; }
    
    .metric-card:hover {
        transform: translateY(-5px) scale(1.05);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
        background: rgba(255,255,255,0.9);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        animation: pulse 2s infinite;
    }
    
    .metric-label {
        font-size: 0.9rem;
        font-weight: 500;
        color: #64748b;
    }
    
    .fake-metric { color: #dc2626; }
    .real-metric { color: #059669; }
    .neutral-metric { color: #3b82f6; }
    
    /* Analysis section */
    .analysis-section {
        background: rgba(255,255,255,0.7);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        animation: slideInLeft 0.6s ease-out 1.5s both;
        backdrop-filter: blur(5px);
        transition: all 0.3s ease;
    }
    
    .analysis-section:hover {
        background: rgba(255,255,255,0.9);
        transform: translateX(5px);
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
        animation: slideInRight 0.5s ease-out both;
        transition: all 0.3s ease;
        backdrop-filter: blur(3px);
    }
    
    .red-flag:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.2);
    }
    
    .credibility-indicator {
        background: rgba(240,253,244,0.8);
        border-left: 4px solid #10b981;
        padding: 1rem;
        margin: 0.8rem 0;
        border-radius: 8px;
        font-weight: 500;
        animation: slideInLeft 0.5s ease-out both;
        transition: all 0.3s ease;
        backdrop-filter: blur(3px);
    }
    
    .credibility-indicator:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2);
    }
    
    /* Text area styling */
    .stTextArea > div > div > textarea {
        width: 100%;
        min-height: 150px;
        padding: 1.5rem;
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        font-size: 1rem;
        font-family: 'Inter', sans-serif;
        line-height: 1.6;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1), 0 4px 20px rgba(102, 126, 234, 0.15);
        outline: none;
        background: #ffffff;
        transform: scale(1.01);
    }
    
    /* Examples section with enhanced animations */
    .examples-section {
        margin: 4rem 2rem;
        text-align: center;
        animation: fadeInUp 1s ease-out 1s both;
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
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        animation: fadeInUp 0.8s ease-out both;
        position: relative;
        overflow: hidden;
    }
    
    .example-card:nth-child(1) { animation-delay: 1.2s; }
    .example-card:nth-child(2) { animation-delay: 1.4s; }
    .example-card:nth-child(3) { animation-delay: 1.6s; }
    
    .example-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.1), transparent);
        transition: left 0.6s ease;
    }
    
    .example-card:hover {
        border-color: #667eea;
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    }
    
    .example-card:hover::before {
        left: 100%;
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
        animation: pulse 2s infinite;
        position: relative;
        overflow: hidden;
    }
    
    .example-type::before {
        content: "";
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: left 0.5s ease;
    }
    
    .example-card:hover .example-type::before {
        left: 100%;
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
        transition: color 0.3s ease;
    }
    
    .example-card:hover .example-text {
        color: #1f2937;
    }
    
    /* Performance stats with enhanced animations */
    .performance-section {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        color: white;
        border-radius: 16px;
        padding: 4rem 3rem;
        margin: 4rem 2rem;
        text-align: center;
        animation: fadeInUp 1s ease-out 1.8s both;
        position: relative;
        overflow: hidden;
    }
    
    .performance-section::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.05), transparent);
        animation: shimmer 8s linear infinite;
        pointer-events: none;
    }
    
    .performance-section .section-title {
        color: white;
        margin-bottom: 3rem;
        animation: fadeInUp 0.8s ease-out 2s both;
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 3rem;
        margin-top: 2rem;
    }
    
    .stat-item {
        text-align: center;
        animation: fadeInUp 0.8s ease-out both;
        transition: transform 0.3s ease;
    }
    
    .stat-item:nth-child(1) { animation-delay: 2.2s; }
    .stat-item:nth-child(2) { animation-delay: 2.4s; }
    .stat-item:nth-child(3) { animation-delay: 2.6s; }
    
    .stat-item:hover {
        transform: translateY(-5px) scale(1.05);
    }
    
    .stat-value {
        font-size: 3rem;
        font-weight: 700;
        color: #10b981;
        display: block;
        margin-bottom: 0.5rem;
        animation: pulse 3s infinite, float 4s ease-in-out infinite;
        text-shadow: 0 4px 8px rgba(16, 185, 129, 0.3);
    }
    
    .stat-item:nth-child(2) .stat-value { animation-delay: 1s; }
    .stat-item:nth-child(3) .stat-value { animation-delay: 2s; }
    
    .stat-label {
        font-size: 1rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Disclaimer with subtle animation */
    .disclaimer {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border: 2px solid #f59e0b;
        border-radius: 16px;
        padding: 2rem;
        margin: 4rem 2rem 2rem 2rem;
        animation: fadeInUp 1s ease-out 2.8s both;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    
    .disclaimer::before {
        content: "";
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(245, 158, 11, 0.1), transparent);
        transition: left 0.6s ease;
    }
    
    .disclaimer:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(245, 158, 11, 0.3);
    }
    
    .disclaimer:hover::before {
        left: 100%;
    }
    
    .disclaimer-title {
        font-weight: 700;
        color: #92400e;
        margin-bottom: 1rem;
        font-size: 1.2rem;
        animation: fadeInUp 0.6s ease-out 3s both;
    }
    
    .disclaimer-text {
        color: #92400e;
        line-height: 1.6;
        animation: fadeInUp 0.6s ease-out 3.2s both;
    }
    
    /* Loading animations */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(255,255,255,.3);
        border-radius: 50%;
        border-top-color: #fff;
        animation: spin 1s ease-in-out infinite;
    }
    
    /* Scroll reveal animations */
    .scroll-reveal {
        opacity: 0;
        transform: translateY(50px);
        transition: all 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .scroll-reveal.revealed {
        opacity: 1;
        transform: translateY(0);
    }
    
    .scroll-reveal-left {
        opacity: 0;
        transform: translateX(-50px);
        transition: all 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .scroll-reveal-left.revealed {
        opacity: 1;
        transform: translateX(0);
    }
    
    .scroll-reveal-right {
        opacity: 0;
        transform: translateX(50px);
        transition: all 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .scroll-reveal-right.revealed {
        opacity: 1;
        transform: translateX(0);
    }
    
    .scroll-reveal-fade {
        opacity: 0;
        transition: all 0.8s ease-out;
    }
    
    .scroll-reveal-fade.revealed {
        opacity: 1;
    }
    
    /* Enhanced hover effects for interactive elements */
    .interactive-element {
        position: relative;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .interactive-element::after {
        content: "";
        position: absolute;
        bottom: 0;
        left: 50%;
        width: 0;
        height: 2px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        transition: all 0.3s ease;
        transform: translateX(-50%);
    }
    
    .interactive-element:hover::after {
        width: 100%;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .animated-cube {
            width: 40px;
            height: 40px;
        }
        
        .cube-face {
            width: 40px;
            height: 40px;
        }
        
        .cube-face.front  { transform: translateZ(20px); }
        .cube-face.back   { transform: rotateY(180deg) translateZ(20px); }
        .cube-face.right  { transform: rotateY(90deg) translateZ(20px); }
        .cube-face.left   { transform: rotateY(-90deg) translateZ(20px); }
        .cube-face.top    { transform: rotateX(90deg) translateZ(20px); }
        .cube-face.bottom { transform: rotateX(-90deg) translateZ(20px); }
        
        .main-title {
            font-size: 2.5rem;
        }
        
        .section-title {
            font-size: 2rem;
        }
        
        .how-it-works,
        .input-section,
        .examples-section,
        .performance-section,
        .disclaimer,
        .result-container {
            margin-left: 1rem;
            margin-right: 1rem;
            padding: 2rem;
        }
        
        .features-grid,
        .examples-grid {
            grid-template-columns: 1fr;
        }
        
        .stats-grid {
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        }
        
        .metrics-grid {
            grid-template-columns: 1fr;
        }
        
        .feature-item,
        .example-card,
        .metric-card {
            animation-delay: 0s !important;
        }
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #7c3aed 0%, #8b5cf6 100%);
    }
</style>
""", unsafe_allow_html=True)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Stopwords load
stop_words = set(stopwords.words('english'))

# ---------- Load model & vectorizer ----------
@st.cache_resource
def load_models():
    try:
        model = joblib.load("model/fake_news_model.pkl")
        vectorizer = joblib.load("model/vectorizer.pkl")
        return model, vectorizer
    except FileNotFoundError:
        st.error("⚠ Model files not found. Please ensure 'fake_news_model.pkl' and 'vectorizer.pkl' are in the 'model/' directory.")
        return None, None

model, vectorizer = load_models()

# ---------- Text Analysis Functions ----------
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def analyze_text_features(text):
    """Analyze various text features for credibility assessment"""
    features = {}
    
    # Basic text statistics
    features['word_count'] = len(text.split())
    features['sentence_count'] = len([s for s in text.split('.') if s.strip()])
    features['char_count'] = len(text)
    
    # Language patterns analysis
    exclamation_count = text.count('!')
    question_count = text.count('?')
    caps_count = sum(1 for c in text if c.isupper())
    
    features['exclamation_ratio'] = exclamation_count / max(features['sentence_count'], 1)
    features['question_ratio'] = question_count / max(features['sentence_count'], 1)
    features['caps_ratio'] = caps_count / max(features['char_count'], 1)
    
    return features

def detect_red_flags(text, features):
    """Detect potential red flags in the text"""
    red_flags = []
    credibility_indicators = []
    
    # Red flags detection
    suspicious_phrases = [
        "big pharma", "doctors hate", "miracle cure", "secret", "they don't want you to know",
        "shocking", "revealed", "exposed", "conspiracy", "cover-up", "banned", "suppressed"
    ]
    
    text_lower = text.lower()
    found_suspicious = [phrase for phrase in suspicious_phrases if phrase in text_lower]
    
    if found_suspicious:
        red_flags.append(f"SUSPICIOUS LANGUAGE: Found '{', '.join(found_suspicious[:3])}'")
    
    if features['exclamation_ratio'] > 0.3:
        red_flags.append("EXCESSIVE PUNCTUATION: Too many exclamation marks may indicate bias")
    
    if features['caps_ratio'] > 0.1:
        red_flags.append("EXCESSIVE CAPS: Overuse of capital letters detected")
    
    # Credibility indicators
    credible_terms = [
        "research", "study", "published", "journal", "university", "professor",
        "data", "analysis", "evidence", "peer-reviewed", "clinical trial"
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
        <div class="animated-cube">
            <div class="cube-face front"></div>
            <div class="cube-face back"></div>
            <div class="cube-face right"></div>
            <div class="cube-face left"></div>
            <div class="cube-face top"></div>
            <div class="cube-face bottom"></div>
        </div>
        <h1 class="main-title">InFactAI</h1>
    </div>
    <p class="tagline">Advanced AI-powered fake news detection</p>
</div>
""", unsafe_allow_html=True)

# ---------- How It Works Section ----------
st.markdown("""
<div class="how-it-works scroll-reveal">
    <h2 class="section-title">How It Works</h2>
    <p class="section-description">
        Our advanced AI analyzes news content using machine learning algorithms trained on 1000+ real and fake news examples. 
        Simply paste any text below for instant analysis.
    </p>
    <div class="features-grid">
        <div class="feature-item scroll-reveal" style="transition-delay: 0.1s;">
            <span class="feature-icon">🤖</span>
            <div class="feature-title">AI-Powered</div>
            <div class="feature-description">Advanced machine learning algorithms for accurate detection</div>
        </div>
        <div class="feature-item scroll-reveal" style="transition-delay: 0.2s;">
            <span class="feature-icon">⚡</span>
            <div class="feature-title">Instant Results</div>
            <div class="feature-description">Get predictions in under 1 second with confidence scores</div>
        </div>
        <div class="feature-item scroll-reveal" style="transition-delay: 0.3s;">
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

# Input option with scroll reveal
st.markdown('<div class="scroll-reveal">', unsafe_allow_html=True)
input_option = st.radio("Choose input method:", 
                       ["Type/Paste Article", "Use Sample Fake News", "Use Sample Real News"])
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="scroll-reveal">', unsafe_allow_html=True)
if input_option == "Use Sample Fake News":
    user_input = st.text_area("", value=sample_fake, height=150, 
                              placeholder="Enter or paste the news article text you want to analyze...",
                              help="Copy and paste the full text of the news article for the most accurate analysis.")
elif input_option == "Use Sample Real News":
    user_input = st.text_area("", value=sample_real, height=150,
                              placeholder="Enter or paste the news article text you want to analyze...",
                              help="Copy and paste the full text of the news article for the most accurate analysis.")
else:
    user_input = st.text_area("", height=150,
                              placeholder="Enter or paste the news article text you want to analyze...",
                              help="Copy and paste the full text of the news article for the most accurate analysis.")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="scroll-reveal">', unsafe_allow_html=True)
analyze_button = st.button("🔍 Analyze Article", key="analyze")
st.markdown('</div>', unsafe_allow_html=True)

if analyze_button:
    if user_input.strip() == "":
        st.warning("⚠ Please enter some text to analyze.")
    elif model is None or vectorizer is None:
        st.error("❌ Model not available. Please check the model files.")
    else:
        # Show loading
        with st.spinner("🔄 Analyzing article..."):
            time.sleep(0.8)  # Brief delay for user experience
            
            # Perform analysis
            cleaned_input = clean_text(user_input)
            transformed_text = vectorizer.transform([cleaned_input])
            prediction = model.predict(transformed_text)[0]
            proba = model.predict_proba(transformed_text)[0]

            # Get probabilities
            fake_prob = proba[1] * 100
            real_prob = proba[0] * 100
            confidence = max(fake_prob, real_prob)
            
            # Analyze text features
            features = analyze_text_features(user_input)
            red_flags, credibility_indicators = detect_red_flags(user_input, features)
            
            # Extract suspicious words
            suspicious_words = []
            for flag in red_flags:
                if "Found '" in flag:
                    start = flag.find("Found '") + 7
                    end = flag.find("'", start)
                    if start > 6 and end > start:
                        words = flag[start:end].split(', ')
                        suspicious_words.extend(words)
                else:
                    if "EXCESSIVE PUNCTUATION" in flag:
                        suspicious_words.append("Too many exclamation marks")
                    elif "EXCESSIVE CAPS" in flag:
                        suspicious_words.append("Overuse of capital letters")
            
            # Extract credible words
            credible_words = []
            for indicator in credibility_indicators:
                if "Found '" in indicator:
                    start = indicator.find("Found '") + 7
                    end = indicator.find("'", start)
                    if start > 6 and end > start:
                        words = indicator[start:end].split(', ')
                        credible_words.extend(words)
                else:
                    if "APPROPRIATE LENGTH" in indicator:
                        credible_words.append("Reasonable article length")

            # Display results
            label = "FAKE" if prediction == 1 else "REAL"
            
            if label == "REAL":
                st.markdown(f"""
                <div class="result-container result-real">
                    <div class="result-title">
                        ✅ REAL NEWS DETECTED
                    </div>
                    <div class="result-description">
                        This article appears to be <strong>authentic</strong> based on our AI analysis.
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-container result-fake">
                    <div class="result-title">
                        ⚠ FAKE NEWS DETECTED
                    </div>
                    <div class="result-description">
                        This article appears to be <strong>potentially misleading</strong> based on our AI analysis.
                    </div>
                """, unsafe_allow_html=True)
                
            # Confidence Section
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
            """, unsafe_allow_html=True)
            
            # Metrics Grid
            st.markdown(f"""
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
            """, unsafe_allow_html=True)
            
            # Detailed Analysis
            st.markdown(f"""
            <div class="analysis-section">
                <div class="analysis-title">🔍 Detailed Analysis</div>
                <p><strong>Text Length:</strong> {features['word_count']} words, {features['sentence_count']} sentences</p>
                <p><strong>Language Patterns:</strong> {len(red_flags)} suspicious indicators, {len(credibility_indicators)} credible indicators</p>
                <p><strong>Analysis Quality:</strong> {'High' if features['word_count'] > 50 else 'Low'}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Red Flags
            if red_flags:
                st.markdown('<div class="analysis-title">🚩 Red Flags Detected:</div>', unsafe_allow_html=True)
                for i, flag in enumerate(red_flags):
                    st.markdown(f'<div class="red-flag" style="animation-delay: {i * 0.1}s">• {flag}</div>', unsafe_allow_html=True)
            
            # Credibility Indicators
            if credibility_indicators:
                st.markdown('<div class="analysis-title">✅ Credibility Indicators:</div>', unsafe_allow_html=True)
                for i, indicator in enumerate(credibility_indicators):
                    st.markdown(f'<div class="credibility-indicator" style="animation-delay: {i * 0.1}s">• {indicator}</div>', unsafe_allow_html=True)
            
            # Suspicious Words
            if suspicious_words:
                st.markdown('<div class="analysis-title">⚠ Suspicious Patterns Found:</div>', unsafe_allow_html=True)
                for i, word in enumerate(suspicious_words[:5]):
                    st.markdown(f'<div class="red-flag" style="animation-delay: {i * 0.1}s">⚠ {word}</div>', unsafe_allow_html=True)
            
            # Credible Words
            if credible_words:
                st.markdown('<div class="analysis-title">✅ Credibility Signs:</div>', unsafe_allow_html=True)
                for i, word in enumerate(credible_words[:5]):
                    st.markdown(f'<div class="credibility-indicator" style="animation-delay: {i * 0.1}s">✅ {word}</div>', unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ---------- Try These Examples Section ----------
st.markdown("""
<div class="examples-section scroll-reveal">
    <h2 class="section-title">Try These Examples</h2>
    <div class="examples-grid">
        <div class="example-card scroll-reveal" style="transition-delay: 0.1s;">
            <div class="example-type real-news">Real News</div>
            <div class="example-text">
                "WATCH TREY GOWDY Crush The Lying Media During Benghazi Report Press Conference ,awesome watched entire press conference gowdy nails lame stream media."
            </div>
        </div>
        <div class="example-card scroll-reveal" style="transition-delay: 0.2s;">
            <div class="example-type fake-news">Fake News</div>
            <div class="example-text">
                "House Speaker Ryan urges coordinated response to Brussels attack,washington reuters us house representatives speaker paul ryan tuesday urged international cooperation reacting series attacks brussels left people dead many injured"
            </div>
        </div>
        <div class="example-card scroll-reveal" style="transition-delay: 0.3s;">
            <div class="example-type health-news">International News</div>
            <div class="example-text">
                "WHAT THE HECK! MICHELLE OBAMA STARS IN AWKWARD KICKBOXING VIDEO,go away bizarre words pictures except ughmichelle obama posted workout video online shows pumping iron crunching abs medicine ball"
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------- Model Performance Section ----------
st.markdown("""
<div class="performance-section scroll-reveal">
    <h2 class="section-title">Model Performance</h2>
    <div class="stats-grid">
        <div class="stat-item scroll-reveal" style="transition-delay: 0.1s;">
            <span class="stat-value">98.9%</span>
            <div class="stat-label">Accuracy</div>
        </div>
        <div class="stat-item scroll-reveal" style="transition-delay: 0.2s;">
            <span class="stat-value">40K+</span>
            <div class="stat-label">Training Data</div>
        </div>
        <div class="stat-item scroll-reveal" style="transition-delay: 0.3s;">
            <span class="stat-value">&lt;1s</span>
            <div class="stat-label">Response Time</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------- Disclaimer Section ----------
st.markdown("""
<div class="disclaimer scroll-reveal">
    <div class="disclaimer-title">⚠ Important Disclaimer</div>
    <div class="disclaimer-text">
        This AI model has been trained on a specific dataset and may not accurately predict the authenticity 
        of news articles that fall outside the scope of its training data. Real-world news content, emerging 
        topics, or articles with novel characteristics may yield inaccurate results. Please use this tool as 
        a preliminary assessment only and always verify information through multiple reliable sources and 
        fact-checking organizations before making any conclusions.
    </div>
</div>
""", unsafe_allow_html=True)

# JavaScript for scroll reveal functionality
st.markdown("""
<script>
document.addEventListener('DOMContentLoaded', function() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('revealed');
            }
        });
    }, observerOptions);

    // Observe all scroll-reveal elements
    const scrollElements = document.querySelectorAll('.scroll-reveal, .scroll-reveal-left, .scroll-reveal-right, .scroll-reveal-fade');
    scrollElements.forEach(el => observer.observe(el));
});
</script>
""", unsafe_allow_html=True)