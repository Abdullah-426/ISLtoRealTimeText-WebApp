# ISL to Real-Time Text WebApp

A comprehensive web application that converts Indian Sign Language (ISL) gestures into real-time text using advanced AI models and computer vision.

## 🚀 Features

- **Real-time Sign Language Recognition**: Converts hand gestures to text in real-time
- **Multi-Modal Support**: Supports both individual letters and complete phrases
- **Client-Side Processing**: Landmark detection runs locally for privacy and speed
- **Ensemble Learning**: Combines multiple AI models for accurate predictions
- **Natural Language Processing**: LLM-powered post-processing for natural text flow
- **Modern UI**: Beautiful dark theme with interactive components

## 🏗️ Architecture

### Frontend (Next.js 14 + TypeScript)
- **Framework**: Next.js 14 with TypeScript and Tailwind CSS
- **Computer Vision**: MediaPipe for hand, pose, and face landmark detection
- **State Management**: Zustand for application state
- **UI Components**: Custom components with shadcn/ui templates
- **Real-time Processing**: Client-side landmark detection and visualization

### Backend Services (FastAPI)
- **Inference Service**: Handles ML model predictions (LSTM, TCN, Letters)
- **Postprocessing Service**: LLM-powered text normalization and enhancement
- **Model Support**: Keras models with custom layers for attention mechanisms

### AI Models
- **Letters Model**: MLP for individual letter recognition (126-D hand landmarks)
- **LSTM Model**: Recurrent neural network for phrase recognition
- **TCN Model**: Temporal Convolutional Network for sequence modeling
- **Ensemble**: Combines multiple models for improved accuracy

## 📁 Project Structure

```
ISLtoRealTimeText App v2/
├── frontend/                 # Next.js frontend application
│   ├── app/                 # Next.js app directory
│   ├── components/          # React components
│   ├── lib/                 # Utility functions and stores
│   └── public/              # Static assets
├── services/                # Backend services
│   ├── infer/              # Model inference service
│   └── postprocess/        # Text postprocessing service
├── models/                  # Trained ML models
│   ├── isl_phrases_v3_lstm/
│   ├── isl_phrases_v3_tcn/
│   └── isl_wcs_raw_aug_light_v2/
├── scripts/                 # Development and deployment scripts
└── infra/                   # Infrastructure configuration
```

## 🛠️ Installation & Setup

### Prerequisites
- Node.js 18+ and npm
- Python 3.8+
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/Abdullah-426/ISLtoRealTimeText-WebApp.git
cd ISLtoRealTimeText-WebApp
```

### 2. Frontend Setup
```bash
cd frontend
npm install
```

### 3. Backend Setup
```bash
# Inference Service
cd services/infer
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Postprocessing Service
cd ../postprocess
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 4. Environment Variables
Create `.env.local` in the frontend directory:
```env
NEXT_PUBLIC_INFER_URL=http://localhost:8001
NEXT_PUBLIC_POSTPROCESS_URL=http://localhost:8000
```

## 🚀 Running the Application

### Development Mode
```bash
# Start all services (Windows)
.\scripts\start-dev.ps1

# Or start individually:
# Terminal 1: Backend Services
cd services/infer && .venv\Scripts\activate && uvicorn main:app --host 0.0.0.0 --port 8001
cd services/postprocess && .venv\Scripts\activate && uvicorn main:app --host 0.0.0.0 --port 8000

# Terminal 2: Frontend
cd frontend && npm run dev
```

### Access the Application
- Frontend: http://localhost:3000
- Inference API: http://localhost:8001
- Postprocessing API: http://localhost:8000

## 🎯 Usage

1. **Open the Application**: Navigate to http://localhost:3000
2. **Enable Camera**: Click "Turn on camera" to start webcam feed
3. **Select Mode**: Choose between "letters", "phrases", or "ensemble" modes
4. **Make Signs**: Perform ISL gestures in front of the camera
5. **View Predictions**: See real-time predictions in the "Top 3" section
6. **Commit Text**: Hold gestures to commit them to the transcript

## 🔧 Technical Details

### Hand Landmark Processing
- **Input**: 126-dimensional feature vector (2 hands × 21 landmarks × 3 coordinates)
- **Preprocessing**: Wrist-center normalization and scale invariance
- **Custom Layers**: Temporal attention mechanisms for sequence modeling

### Model Architecture
- **Letters**: Multi-layer perceptron with custom preprocessing functions
- **LSTM**: Bidirectional LSTM with attention mechanism
- **TCN**: Temporal Convolutional Network with residual connections

### Real-time Processing
- **MediaPipe**: Hand, pose, and face landmark detection
- **Smoothing**: Exponential moving average for stable predictions
- **Commit Logic**: Hold-to-commit with confidence thresholds

## 📊 Model Performance

- **Letters Recognition**: High accuracy for individual letter signs
- **Phrase Recognition**: Context-aware sequence modeling
- **Real-time Processing**: <100ms inference latency
- **Robustness**: Handles various lighting and background conditions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Creators

- **Abdullah Ansari** - Lead Developer
- **Aryan Tayal** - Developer
- **Pranav Bansal** - Developer
- **Devyansh Kirar** - Developer

## 🙏 Acknowledgments

- MediaPipe team for computer vision capabilities
- TensorFlow team for ML framework
- Next.js team for the amazing React framework
- All contributors and testers

## 📞 Support

If you encounter any issues or have questions, please open an issue on GitHub or contact the development team.

---

**Note**: This application requires a webcam and works best with good lighting conditions. Make sure to allow camera permissions when prompted.
