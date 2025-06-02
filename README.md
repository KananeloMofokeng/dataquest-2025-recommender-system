# DataQuest 2025 – Customer Recommender System

This repository contains my submission for the DataQuest 2025 Recommender Systems Challenge, where I developed a customer-centric recommendation engine using behavioral clustering and collaborative filtering.

## 📌 Project Overview

The goal was to build a recommender system that could personalize product offers based on user behavior patterns derived from interaction data.

### Key Features
- Cleaned and transformed 400K+ raw interaction records.
- Engineered behavioral features (recency, click-to-checkout ratio, item diversity, etc.).
- Used PCA for dimensionality reduction (85% variance retained).
- Applied KMeans to segment customers into behavioral clusters.
- Built a cluster-aware recommender system with fallback support.
- Evaluated clusters with Silhouette, Calinski-Harabasz, and Davies-Bouldin scores.

## 🔧 Setup

```bash
git clone https://github.com/yourusername/dataquest-2025-recommender-system.git
cd dataquest-2025-recommender-system
pip install -r requirements.txt
