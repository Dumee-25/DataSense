# DataSense

**AI-powered data analysis and consulting platform.**
Upload a CSV. Get expert-level structural, statistical, and ML-ready insights — enhanced by LLMs — in seconds.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-14-000000?logo=next.js&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-4169E1?logo=postgresql&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Database Setup](#database-setup)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
- [Configuration](#configuration)
  - [Environment Variables](#environment-variables)
  - [LLM Provider Setup](#llm-provider-setup)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)

---

## Overview

DataSense is a full-stack data analysis platform that turns raw CSV files into actionable, expert-level reports. It combines classical statistical analysis with LLM-enhanced insights to deliver findings that are understandable by both technical and non-technical stakeholders.

**Core flow:**

```
Upload CSV → Validate → Structural Analysis → Statistical Engine
           → Model Recommendations → LLM Insights → Results Dashboard / PDF Export
```

The entire analysis runs as a 7-step background pipeline with real-time progress tracking, so users can watch each stage complete live in the browser.

---

## Features

### Data Analysis
- **Structural profiling** — column types, cardinality, missing values, duplicates, memory usage, data structure detection (time-series, panel, cross-sectional)
- **Quality auditing** — sentinel value scrubbing, disguised missing values, whitespace issues, mixed types, constant columns, infinite values
- **Statistical engine** — correlation analysis, outlier detection (IQR-based), distribution profiling (skewness, kurtosis), Simpson's paradox detection, multicollinearity warnings, heteroscedasticity checks, class imbalance detection
- **ML model recommendations** — scores and ranks classification/regression models based on dataset characteristics, with confidence scores, preprocessing steps, cross-validation strategy, and evaluation metrics
- **Target column suggestion** — heuristic scoring to identify likely prediction targets

### AI / LLM Integration
- **Multi-provider support** — Ollama (local, default), OpenAI, or Groq
- **Dataset context inference** — domain, purpose, column meanings, key risks, leakage suspects
- **Persona system** — tailor insight language for different audiences: `general`, `executive`, `data_scientist`, `product_manager`
- **Prompt caching** — SHA-256 keyed cache to avoid redundant LLM calls
- **Graceful degradation** — halves token budget after consecutive timeouts; falls back to rule-based insights if LLM is unavailable

### Platform
- **Real-time progress** — 7-step pipeline with live percentage updates and step checklist
- **PDF report export** — professionally designed reports with cover page, severity-coded findings, model recommendations, column profiles, and custom icon illustrations
- **History** — browse past analyses with metadata (rows, columns, issues, primary model)
- **Authentication** — optional accounts with seamless anonymous-to-authenticated transition (existing analysis history is preserved on sign-up/login)
- **Session-based tracking** — anonymous users get a browser session cookie; analyses are tied to sessions and optionally linked to accounts

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Frontend (Next.js 14 + React 18 + Tailwind CSS)        │
│  ┌──────────┐ ┌──────────┐ ┌─────────┐ ┌───────────┐   │
│  │  Upload   │ │ Analyzing│ │ Results │ │  History   │   │
│  │  Page     │ │ Progress │ │Dashboard│ │   Page     │   │
│  └──────────┘ └──────────┘ └─────────┘ └───────────┘   │
│                        │ /api proxy                      │
└────────────────────────┼────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Backend (FastAPI + Python)                              │
│  ┌────────────┐  ┌──────────────────────────────────┐   │
│  │  Auth       │  │  Analysis Pipeline (Background)   │  │
│  │  Routes     │  │  1. Validate file                 │  │
│  │            │  │  2. Load dataset                   │  │
│  │  Analysis  │  │  3. Structural analysis            │  │
│  │  Routes    │  │  4. Statistical engine             │  │
│  │            │  │  5. Model recommendations          │  │
│  │  PDF Export│  │  6. LLM-enhanced insights          │  │
│  └────────────┘  │  7. Save results                   │  │
│                   └──────────────────────────────────┘   │
│                        │                                 │
└────────────────────────┼─────────────────────────────────┘
                         ▼
┌──────────────────┐  ┌────────────────────────────────────┐
│  PostgreSQL      │  │  LLM Provider                      │
│  UUID PKs        │  │  • Ollama (local, default)          │
│  JSONB results   │  │  • OpenAI (gpt-4o-mini)            │
│                  │  │  • Groq (llama-3.1-8b-instant)     │
└──────────────────┘  └────────────────────────────────────┘
```

---

## Tech Stack

| Layer        | Technology                                              |
| ------------ | ------------------------------------------------------- |
| **Frontend** | Next.js 14, React 18, TypeScript 5, Tailwind CSS 3      |
| **Backend**  | FastAPI 0.115, Python 3.11+, SQLAlchemy 2.0, Alembic    |
| **Database** | PostgreSQL (UUID primary keys, JSONB for analysis results) |
| **AI / LLM** | Ollama (local), OpenAI, or Groq — configurable          |
| **Analysis** | Pandas 2.2, SciPy 1.13, scikit-learn 1.5               |
| **PDF**      | ReportLab 4.2                                           |
| **Auth**     | JWT (python-jose) + bcrypt (passlib), httponly cookies   |

---

## Getting Started

### Prerequisites

- **Python 3.11+**
- **Node.js 18+** and npm
- **PostgreSQL 15+**
- **Ollama** (optional, for local LLM insights) — [Install Ollama](https://ollama.com)

### Database Setup

1. Create the PostgreSQL database and user:

```sql
CREATE USER datasense_user WITH PASSWORD 'datasense123';
CREATE DATABASE datasense OWNER datasense_user;
```

2. Run migrations:

```bash
cd backend
alembic upgrade head
```

### Backend Setup

```bash
cd backend

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn api.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. Health check: `GET http://localhost:8000/health`.

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

The app will be available at `http://localhost:3000`. API calls are proxied to the backend via Next.js rewrites.

---

## Configuration

### Environment Variables

Create a `.env` file in the `backend/` directory:

```env
# Database
DATABASE_URL=postgresql://datasense_user:datasense123@localhost:5432/datasense

# Authentication
SECRET_KEY=your-secret-key-here

# LLM Provider: "ollama" (default), "openai", or "groq"
LLM_PROVIDER=ollama

# Ollama (local) — default
OLLAMA_URL=http://localhost:11434

# OpenAI (optional)
OPENAI_API_KEY=sk-...

# Groq (optional)
GROQ_API_KEY=gsk_...

# LLM Tuning
LLM_TIMEOUT=30
LLM_MAX_WORKERS=3
TOKEN_BUDGET=4096

# Insight persona: "general" (default), "executive", "data_scientist", "product_manager"
INSIGHT_PERSONA=general

# Analysis toggle — set to false to skip LLM calls and use rule-based insights only
USE_LLM=true
```

### LLM Provider Setup

| Provider   | Setup                                                                 |
| ---------- | --------------------------------------------------------------------- |
| **Ollama** | Install Ollama, then run `ollama pull llama3.2:latest`                |
| **OpenAI** | Set `LLM_PROVIDER=openai` and provide `OPENAI_API_KEY`               |
| **Groq**   | Set `LLM_PROVIDER=groq` and provide `GROQ_API_KEY`                   |

If no LLM provider is available, DataSense falls back gracefully to rule-based insights.

---

## Usage

1. **Upload** — Drag and drop a CSV file (up to 50 MB) on the home page.
2. **Analyze** — Watch the 7-step pipeline process your data in real time.
3. **Explore results** — Browse findings across five tabs:
   - **Summary** — executive overview, quick wins, severity breakdown
   - **Findings** — expandable cards with severity, business impact, and AI deep dives
   - **Relations** — column correlations, Simpson's paradox, class imbalance guidance
   - **Model** — recommended ML model with confidence, alternatives, and preprocessing steps
   - **Columns** — detailed profile of every column
4. **Export** — Download a professionally formatted PDF report.
5. **History** — Revisit past analyses from the History page. Sign in to preserve history across devices.

---

## API Reference

### Analysis

| Method   | Endpoint                          | Description                          |
| -------- | --------------------------------- | ------------------------------------ |
| `POST`   | `/api/analyze`                    | Upload CSV and start analysis        |
| `GET`    | `/api/status/{job_id}`            | Poll job progress (0–100%)           |
| `GET`    | `/api/results/{job_id}`           | Fetch full analysis results          |
| `DELETE` | `/api/cancel/{job_id}`            | Cancel a queued or running job       |
| `GET`    | `/api/jobs`                       | List jobs for the current session    |
| `DELETE` | `/api/jobs/{job_id}`              | Delete a specific job and results    |
| `DELETE` | `/api/jobs`                       | Cleanup expired jobs                 |
| `GET`    | `/api/results/{job_id}/export/pdf`| Download PDF report                  |

### Authentication

| Method   | Endpoint              | Description                          |
| -------- | --------------------- | ------------------------------------ |
| `POST`   | `/api/auth/register`  | Create an account                    |
| `POST`   | `/api/auth/login`     | Sign in                              |
| `POST`   | `/api/auth/logout`    | Sign out                             |
| `GET`    | `/api/auth/me`        | Get current user profile             |
| `PATCH`  | `/api/auth/me`        | Update name or password              |
| `DELETE` | `/api/auth/me`        | Delete account and all data          |

---

## Project Structure

```
datasense/
├── backend/
│   ├── api/
│   │   ├── main.py              # FastAPI app, CORS, lifespan, route mounting
│   │   ├── routes.py            # Analysis endpoints, background pipeline
│   │   └── auth_routes.py       # Authentication endpoints
│   ├── core/
│   │   ├── structural_analyzer.py   # Dataset structure & quality profiling
│   │   ├── statistical_engine.py    # Correlations, outliers, distributions, red flags
│   │   ├── model_recommender.py     # ML model scoring & recommendations
│   │   ├── insight_generator.py     # LLM-enhanced insight engine
│   │   └── pdf_generator.py         # ReportLab PDF report generation
│   ├── database/
│   │   ├── connection.py        # SQLAlchemy engine & session
│   │   ├── models.py            # User, Session, Job, Result, DatasetMetadata
│   │   ├── crud.py              # Job & result CRUD operations
│   │   ├── auth_crud.py         # User & session CRUD operations
│   │   └── migrations/          # Alembic migrations
│   ├── utils/
│   │   ├── auth.py              # JWT & bcrypt helpers
│   │   ├── data_validator.py    # Pre-pipeline file validation
│   │   └── dependencies.py      # FastAPI dependency injection
│   ├── requirements.txt
│   └── alembic.ini
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx             # Upload / landing page
│   │   │   ├── layout.tsx           # Root layout
│   │   │   ├── globals.css          # Global styles
│   │   │   ├── analyzing/[jobId]/   # Real-time progress page
│   │   │   ├── results/[jobId]/     # Results dashboard (5 tabs)
│   │   │   └── history/             # Analysis history
│   │   ├── components/
│   │   │   ├── NavBar.tsx           # Top navigation bar
│   │   │   └── AuthModal.tsx        # Login / register modal
│   │   └── lib/
│   │       └── datasense-api.ts     # Typed API client
│   ├── next.config.js               # API proxy rewrites
│   ├── tailwind.config.js
│   ├── tsconfig.json
│   └── package.json
└── README.md
```