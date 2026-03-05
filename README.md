# DataSense

**AI-powered data analysis platform.**
Upload a CSV. Get clear, actionable insights about your data — what's good, what needs fixing, and what model to use — in seconds.

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

DataSense is a data analysis platform that turns raw CSV files into clear, actionable reports. It runs standard statistical checks on your data, then uses an AI model to explain the findings in plain language anyone can understand.

**Core flow:**

```
Upload CSV → Validate → Structural Analysis → Statistical Engine
           → Model Recommendations → LLM Insights → Results Dashboard / PDF Export
```

The entire analysis runs as a 7-step background pipeline with real-time progress tracking, so users can watch each stage complete live in the browser.

---

## Features

### Data Analysis
- **Structural profiling** — detects column types, missing values, duplicates, memory usage, and whether your data is time-series, panel, or a regular table
- **Quality checks** — finds hidden missing values, whitespace problems, mixed data types, constant columns, and invalid entries
- **Statistical analysis** — checks which columns are related, finds unusual values (outliers), looks at how data is spread out, detects when patterns reverse in subgroups, and flags when your target column is lopsided
- **Model recommendations** — suggests the best ML model for your data, with a confidence score, data prep steps, how to test it, and what success looks like
- **Target column detection** — automatically identifies which column you're most likely trying to predict

### AI Integration
- **Multiple AI providers** — works with Ollama (local, default), OpenAI, or Groq
- **Dataset understanding** — the AI figures out what your data is about, what the columns mean, and what domain-specific risks to watch for
- **Audience modes** — switch between `general` (plain language), `executive` (business focus), `data_scientist` (technical), or `product_manager` (product-oriented)
- **Response caching** — remembers previous AI answers so identical analyses don't re-run
- **Works without AI** — if no AI provider is available, you still get rule-based insights; if the AI is slow, it automatically adjusts to keep things moving

### Charts & Visualization
- **14 auto-generated charts** — built from pre-computed results, adapts layout for wide datasets
  - **Data Health Radar** — overall data quality scores at a glance
  - **Missing Values** — shows which columns have gaps and how much
  - **Correlation** — heatmap (small datasets) or bar chart (wide datasets) showing which columns are related
  - **Target Distribution** — how balanced your target column is
  - **Outlier Severity** — which columns have unusual values and how many
  - **Skewness** — how lopsided each column's data distribution is
  - **Column Types** — donut chart showing the breakdown of data types across columns
  - **Duplicate Rows** — unique vs duplicate row counts
  - **Cardinality** — unique value counts per column, helps spot IDs and low-variance columns
  - **Feature Importance** — estimated importance of each feature based on correlation with the target
  - **Missing Data Pattern** — sorted overview of all columns that have gaps
  - **Box Plots** — spread and outliers for the top numeric columns
  - **PCA Variance** — how much variance each principal component explains
  - **Cluster Preview** — data projected onto first two principal components, coloured by auto-detected cluster
- **Smart display** — charts that have no relevant data are automatically hidden
- **Downloadable** — each chart can be downloaded as a PNG image

### Security & Operations
- **Rate limiting** — per-IP sliding-window rate limiter with two tiers: global (60 req/min) and strict (10 req/min) for upload & auth endpoints; returns `429` with `Retry-After` header
- **File content validation** — rejects binary files (PNG, JPEG, ZIP, PDF, etc.), validates CSV parseability, enforces a 2 000-column limit, and warns on formula-injection patterns (`=`, `+`, `-`, `@`, `|`)
- **Structured logging** — rotating file logs (`logs/datasense.log`, 5 MB × 5 backups) plus formatted console output; configurable via environment variables
- **Concurrency control** — thread-safe active-job counter with configurable max concurrent jobs

### Platform
- **Real-time progress** — 7-step pipeline with live progress bar and step checklist
- **PDF report export** — downloadable reports with findings, model recommendations, column details, and charts
- **History** — browse past analyses with key info (rows, columns, issues, model used)
- **Authentication** — optional accounts; your analysis history transfers when you sign up
- **Works without an account** — anonymous users get a session cookie so their analyses are saved

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
│  │  Routes     │  │  1. Check file                     │  │
│  │            │  │  2. Load data                      │  │
│  │  Analysis  │  │  3. Examine structure              │  │
│  │  Routes    │  │  4. Run statistics                 │  │
│  │            │  │  5. Pick best model                │  │
│  │  PDF Export│  │  6. AI-powered explanations        │  │
│  └────────────┘  │  7. Save results                   │  │
│                   └──────────────────────────────────┘   │
│                        │                                 │
└────────────────────────┼─────────────────────────────────┘
                         ▼
┌──────────────────┐  ┌────────────────────────────────────┐
│  PostgreSQL      │  │  AI Provider                        │
│  Stores results  │  │  • Ollama (local, default)          │
│  as JSON         │  │  • OpenAI (gpt-4o-mini)            │
│                  │  │  • Groq (llama-3.1-8b-instant)     │
└──────────────────┘  └────────────────────────────────────┘
```

---

## Tech Stack

| Layer        | Technology                                              |
| ------------ | ------------------------------------------------------- |
| **Frontend** | Next.js 14, React 18, TypeScript 5, Tailwind CSS 3      |
| **Backend**  | FastAPI 0.115, Python 3.11+, SQLAlchemy 2.0, Alembic    |
| **Database** | PostgreSQL (stores results as structured JSON)              |
| **AI**       | Ollama (local), OpenAI, or Groq — configurable          |
| **Analysis** | Pandas 2.2, SciPy 1.13, scikit-learn 1.5, NumPy 1.26   |
| **Charts**   | Matplotlib 3.10                                         |
| **PDF**      | ReportLab 4.2                                           |
| **Auth**     | JWT + bcrypt, httponly cookies                           |

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

# Ollama (local) — default (must include /api/generate)
OLLAMA_URL=http://localhost:11434/api/generate

# OpenAI (optional)
OPENAI_API_KEY=sk-...

# Groq (optional)
GROQ_API_KEY=gsk_...

# LLM Tuning
LLM_TIMEOUT=60
LLM_BATCH_TIMEOUT=90
LLM_MAX_WORKERS=2
TOKEN_BUDGET=4096

# Insight persona: "general" (default), "executive", "data_scientist", "product_manager"
INSIGHT_PERSONA=general

# Set to false to skip AI calls and use rule-based insights only
USE_LLM=true

# Upload & Concurrency
MAX_FILE_SIZE_MB=50
MAX_CONCURRENT_JOBS=10
JOB_RETENTION_HOURS=48
TEMP_UPLOAD_DIR=temp_uploads

# Rate Limiting
RATE_LIMIT_REQUESTS=60          # Global: max requests per window
RATE_LIMIT_WINDOW_S=60          # Global: window in seconds
RATE_LIMIT_STRICT_REQUESTS=10   # Strict (upload/auth): max per window
RATE_LIMIT_STRICT_WINDOW_S=60   # Strict: window in seconds

# Logging
LOG_LEVEL=INFO
LOG_DIR=logs
LOG_FILE=datasense.log
LOG_MAX_BYTES=5242880           # 5 MB
LOG_BACKUP_COUNT=5
```

### LLM Provider Setup

| Provider   | Setup                                                                 |
| ---------- | --------------------------------------------------------------------- |
| **Ollama** | Install Ollama, then run `ollama pull llama3.2:latest`                |
| **OpenAI** | Set `LLM_PROVIDER=openai` and provide `OPENAI_API_KEY`               |
| **Groq**   | Set `LLM_PROVIDER=groq` and provide `GROQ_API_KEY`                   |

If no AI provider is available, DataSense still works — you get rule-based insights without the AI explanations.

---

## Usage

1. **Upload** — Drag and drop a CSV file (up to 50 MB) on the home page. The file is validated for correct encoding, CSV structure, and content before processing.
2. **Analyze** — Watch the 7-step pipeline process your data in real time.
3. **Explore results** — Browse findings across six tabs:
   - **Summary** — overview of your data, key story, quick wins, and how many issues were found
   - **Findings** — expandable cards for each issue with what's wrong, why it matters, and how to fix it
   - **Charts** — 14 visual charts covering data health, missing values, correlations, target balance, outliers, skewness, column types, duplicates, cardinality, feature importance, missing patterns, box plots, PCA variance, and cluster preview
   - **Relations** — how columns relate to each other, plus guidance on class imbalance if relevant
   - **Model** — which ML model to use, why, alternatives, data prep steps, and how to test it
   - **Columns** — profile of every column (type, missing %, unique values, notes)
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
| `GET`    | `/api/results/{job_id}/charts`    | Fetch base64-encoded chart PNGs      |

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
│   │   ├── main.py              # App setup, CORS, rate limiter, startup hooks
│   │   ├── routes.py            # Analysis endpoints, background pipeline
│   │   └── auth_routes.py       # Login/register endpoints
│   ├── core/
│   │   ├── structural_analyzer.py   # Detects column types, missing data, duplicates
│   │   ├── statistical_engine.py    # Finds correlations, outliers, data patterns, PCA, clustering
│   │   ├── model_recommender.py     # Picks the best ML model for the data
│   │   ├── insight_generator.py     # AI-powered plain-language explanations
│   │   ├── deterministic_summary.py # Prepares computed facts for the AI (no raw data sent)
│   │   ├── aggregation_engine.py    # Groups similar findings together
│   │   ├── relevance_filter.py      # Adjusts findings based on the recommended model
│   │   ├── chart_engine.py          # Generates the 14 analysis charts
│   │   └── pdf_generator.py         # Creates downloadable PDF reports
│   ├── database/
│   │   ├── connection.py        # Database connection setup
│   │   ├── models.py            # Database tables: users, sessions, jobs, results
│   │   ├── crud.py              # Job & result create/read/update/delete
│   │   ├── auth_crud.py         # User & session create/read/update/delete
│   │   └── migrations/          # Database schema migrations
│   ├── utils/
│   │   ├── auth.py              # Login token & password helpers
│   │   ├── data_validator.py    # Validates uploads (binary, CSV parse, formula injection)
│   │   ├── rate_limiter.py      # Per-IP sliding-window rate limiter middleware
│   │   ├── logging_config.py    # Rotating file + console log setup
│   │   └── dependencies.py      # Shared request helpers
│   ├── requirements.txt
│   └── alembic.ini
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx             # Upload / landing page
│   │   │   ├── layout.tsx           # Root layout
│   │   │   ├── globals.css          # Global styles
│   │   │   ├── analyzing/[jobId]/   # Live progress page
│   │   │   ├── results/[jobId]/     # Results dashboard (6 tabs)
│   │   │   └── history/             # Past analyses
│   │   ├── components/
│   │   │   ├── NavBar.tsx           # Top navigation bar
│   │   │   └── AuthModal.tsx        # Login / register modal
│   │   └── lib/
│   │       └── datasense-api.ts     # Typed API client
│   ├── next.config.js               # Routes API calls to the backend
│   ├── tailwind.config.js
│   ├── tsconfig.json
│   └── package.json
└── README.md
```