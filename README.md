# Fraud Detection Monitoring System

## Introduction
The Fraud Detection Monitoring System is a real-time application designed to detect fraudulent transactions. It leverages machine learning to classify transactions as fraudulent or non-fraudulent, while providing a clean and simple web-based interface for monitoring. This project demonstrates the practical implementation of machine learning models in a real-world scenario with real-time data streaming and visualization.

# Features
## Real-time Monitoring:
- Streams incoming transactions and displays them on the dashboard.
- Highlights fraudulent transactions for quick identification.

## Web-based Dashboard:
- Displays all transactions and fraud-specific transactions.
- Dynamically updates using WebSocket technology.

## Machine Learning:
- A Random Forest classifier trained on the credit card fraud dataset.
- Handles class imbalance using SMOTEENN (Synthetic Minority Over-sampling Technique + Edited Nearest Neighbors).

## Simulation:
- Simulates real-time streaming of transactions from the dataset.

## Logging:
- Logs predictions and simulation results for audit and analysis.

# Technologies Used
- Backend: Flask, Flask-SocketIO
- Frontend: HTML, CSS, JavaScript (with Socket.IO for real-time updates)
- Machine Learning: Scikit-learn, Imbalanced-learn, Joblib
- Data Processing: Pandas
- Real-time Streaming: WebSockets

# Architecture overview:
1. Data ingestion:
- A simulate_live_data.py script sends transaction data to the backend in real time.

2. backend processing:
- flask receives data via REST API endpoint
- the machine learnign model classifies transactions.
- predictions are logged and emitted to the frontend via WebSocket.

3. Frontend Dashboard:
- Displays all transactions and fraudulent transactions in seperate sections.
- Automatically updates using WebSocket events.

4. Logging:
- tracks predictions in predictions.log
- tracks simulation activity in simulation.log