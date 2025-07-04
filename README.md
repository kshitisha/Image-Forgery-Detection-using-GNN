

# Image Forgery Detection 

This application uses a Graph Neural Network (GNN) to detect and visualize forgery in images.

<pre> project-root/
├── public/                     # Static frontend files (images, icons, etc.)

├── src/                        # Application source code
│   ├── components/             # Reusable React components (e.g., Navbar, Card)
│   ├── pages/                  # React pages (e.g., Home.jsx, Upload.jsx)
│   ├── server/                 # Backend (Node.js + Python)
│   │   ├── analyze_image.py    # Python script for image analysis
│   │   ├── server.js           # Express server entry point
│   │   └── package.json        # Backend dependencies
│   └── uploads/                # Temp folder for uploaded images

├── gnn_model.pth               # Trained GNN model (you need to provide this)
└── package.json                # Frontend dependencies
  </pre>





## Prerequisites

1. Node.js (v14 or higher)
2. Python 3.7+ with the following packages:
   - torch
   - torchvision
   - torch_geometric
   - opencv-python
   - scikit-image
   - matplotlib
   - networkx
   - numpy

## Setup Instructions

### 1. Install Frontend Dependencies

bash
npm install


### 2. Install Backend Dependencies

bash
cd src/server
npm install


### 3. Install Python Dependencies

bash
pip install torch torchvision torch_geometric opencv-python scikit-image matplotlib networkx numpy


### 4. Place your trained model

Make sure your trained GNN model (gnn_model.pth) is in the root directory of the project.

## Running the Application

### 1. Start the Backend Server

bash
cd src/server
npm start


The server will start on port 5000 by default.

### 2. Start the Frontend Development Server

In a new terminal:

bash
npm run dev


The application will be available at http://localhost:3000

## API Endpoints

- POST /api/analyze: Upload and analyze an image for forgery

## Environment Variables

- PORT: Server port (default: 5000)
- VITE_API_URL: API URL for the frontend to connect to the backend

## Development

- Backend: Express.js server with multer for file uploads
- Frontend: React with Tailwind CSS and shadcn/ui components
- Machine Learning: Python with PyTorch, torch_geometric, and OpenCV

## Troubleshooting

- Make sure the Python script has executable permission: chmod +x src/server/analyze_image.py
- Verify that the model file path in analyze_image.py is correct
- Check server logs for any Python execution errors
