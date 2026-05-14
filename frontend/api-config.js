/**
 * API Configuration for NeuroMed AI Dashboard
 * 
 * Replace the placeholders below with your actual credentials.
 * This file is used by clinical-ai-dashboard.jsx to connect to Firebase and MongoDB/Backend.
 */

export const FIREBASE_CONFIG = {
  apiKey: "YOUR_FIREBASE_API_KEY",
  authDomain: "YOUR_PROJECT.firebaseapp.com",
  projectId: "YOUR_PROJECT_ID",
  storageBucket: "YOUR_PROJECT.appspot.com",
  messagingSenderId: "YOUR_SENDER_ID",
  appId: "YOUR_APP_ID"
};

// If using a backend (FastAPI) that connects to MongoDB:
export const BACKEND_API_URL = "http://localhost:8001"; 

// If you are using a direct MongoDB API (e.g. MongoDB Atlas App Services):
export const MONGODB_CONFIG = {
  dataApiUrl: "YOUR_MONGODB_DATA_API_ENDPOINT",
  apiKey: "YOUR_MONGODB_DATA_API_KEY",
  dataSource: "Cluster0",
  database: "neuromed_db",
  collection: "patients"
};

export const USE_LIVE_DATA = false; // Set to true to enable API calls, false to use dummy data only
