# 📰 News Scraper API  
**Module:** News Bias Analysis Pipeline  

---

## 1. Overview

The **News Scraper API** is responsible for collecting news articles from provided URLs. It serves as the primary data ingestion component in the News Bias Analysis pipeline.

This API extracts structured article data including:

- Headline  
- Article body  
- Source name  
- URL  

The extracted content is passed to downstream components for preprocessing and bias scoring.

---

## 2. Evolution of the Scraper

### Phase 1: Custom Source-Specific Scrapers

During initial data collection, we built custom scrapers for ~10 major Indian news sources, including:

- The Scroll  
- Hindustan Times  
- India Today  
- The Indian Express  

These scrapers were built to:

- Handle different HTML structures  
- Bypass dynamic content loading issues  
- Extract clean article text with minimal noise  
- Improve scraping reliability for high-priority sources  

This phase was specifically designed for large-scale dataset creation.


### Phase 2: Generalized Scraper (Current Production Version)

The API now contains a generalized scraping engine that:

- Supports ~90% of sources in our curated list  
- Automatically detects article structure  
- Handles most static and semi-dynamic news pages  
- Reduces the need for source-specific logic  

---

## 3. Architecture Role in Pipeline




<table>
  <tr>
    <td align="center"><strong>Input Article URL</strong></td>
  </tr>
  <tr>
    <td align="center">↓</td>
  </tr>
  <tr>
    <td align="center"><strong>News Scraper API</strong><br>
    <em>Content Extraction Engine</em></td>
  </tr>
  <tr>
    <td align="center">↓</td>
  </tr>
  <tr>
    <td align="center"><strong>Text Cleaning & Normalization</strong><br>
    <em>Noise Removal • Formatting • Validation</em></td>
  </tr>
  <tr>
    <td align="center">↓</td>
  </tr>
  <tr>
    <td align="center"><strong>Bias Analysis Pipeline</strong></td>
  </tr>
</table>


This API acts as the **data ingestion layer** of the News Bias Analysis system.

---

## 4. Key Features

- Source detection  
- Clean text extraction  
- Extensible scraper architecture  

---

## 5. Setup Instructions

### Step 1: Set Up Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate   # Mac/Linux
```
### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

Common libraries:

- requests
- beautifulsoup4
- newspaper3k (if used)
- lxml
- fastapi


### Step 3: Run the API
If using FastAPI:
```bash

uvicorn main:app --reload
```
If using Flask:
```bash
python app.py
```

---

6. Limitations

- Does not handle highly dynamic JavaScript-heavy pages without headless browser
- May fail on paywalled content
- Requires periodic updates when source HTML changes

---

7. Future Improvements

- Headless browser integration (Selenium / Playwright)
- Automatic layout detection via ML
- Source reliability scoring
- Duplicate article detection

