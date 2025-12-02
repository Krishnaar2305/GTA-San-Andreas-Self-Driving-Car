 # 🚗 Vision-Based Autonomous Driving Agent for GTA San Andreas

A fully vision-driven autonomous driving agent for **Grand Theft Auto: San Andreas**, combining modern deep learning (YOLOPv2 + YOLOv12), lane segmentation, object detection, real-time threat evaluation, and PID-based steering control through a virtual joystick driver (vJoy).  

This project is designed for research, AI experimentation, and autonomous agent development inside open-world game environments.

---

## 📺 Demo Videos  
Watch real driving demos:  
https://drive.google.com/drive/folders/1Ckx4I6JZmcKh-9zZq5jFQRSUI5LXWXg4?usp=sharing

---

# 🚀 Key Features

### 🔵 Lane Keeping Assist (LKA)
- Uses **YOLOPv2** segmentation output.
- Computes curvature, lane center deviation, and steering error.
- Smooths the signal using an **Exponential Moving Average (EMA)**.
- Stabilizes steering using a **PID controller**.

### 🔴 Object Detection & Tracking (YOLOv12)
- Detects vehicles, pedestrians, obstacles.
- Tracks bounding boxes across frames.
- Computes **growth rate** of object area to estimate Time-To-Collision (TTC).
- Produces a **threat priority score**:
  ```
  (growth_rate^2) × (area_fraction)
  ```

### ⚠️ Collision Avoidance System
- **Adaptive Cruise Control (ACC)** → Slow down for moderate threats.
- **Evasive Steering** → Hard steer left/right based on obstacle location.
- **AEB (Automatic Emergency Braking)** → Triggered for critical TTC events.

### 🎮 Virtual Joystick Integration (vJoy + pyvjoy)
- Simulates an Xbox 360 controller.
- Steering and throttle mapped to 0–32768.
- Neutral steering = **16384**.

### 🖥️ Additional Engineering
- Real-time colored logs (via `colorama`).
- A process manager (`path_finder.py`) that ensures correct working paths.
- GPU-accelerated inference pipeline.

---

# 🛠️ System Requirements

| Component | Requirement |
|----------|-------------|
| OS | Windows 10/11 (required for vJoy) |
| Hardware | NVIDIA GPU recommended |
| Game | GTA San Andreas (Windowed/Borderless required) |
| Python | 3.8+ |
| Tools | vJoy, CUDA, PyTorch, Ultralytics |

---

# 📦 Installation

## 1️⃣ Install Dependencies  

Create `requirements.txt`:

```
torch==2.7.1+cu118
opencv-python==4.12.0.88
numpy==2.0.2
pyvjoy==1.0.1
keyboard==0.13.5
ultralytics==8.3.226
colorama==0.4.6
```

Install:

```
pip install -r requirements.txt
```

---

## 2️⃣ Install & Configure vJoy

1. Install **vJoy**  
2. Open **Configure vJoy**  
3. Enable **Device 1**  
4. Enable axes → **X, Y, Z, Rz**  
5. Apply settings  

---

## 3️⃣ Setup Deep Learning Models

### YOLOPv2 (for Lane Segmentation)

Source:  
https://github.com/CAIC-AD/YOLOPv2.git

**Required:**
- Copy the entire `utils/` folder from YOLOPv2 → place in your project root.
- Download the YOLOPv2 weights:

```
/project_root/data/weights/yolopv2.pt
```

---

### YOLOv12 (for Object Detection)

Download your YOLOv12 model (`yolo12n.pt`) and place it here:

```
/project_root/yolo12n.pt
```

---

# 📁 Directory Structure

```
/project_root
│
├── main.py
├── pid_controller.py
├── path_finder.py
├── yolo12n.pt
├── demo_videos/
├── data/
│   └── weights/
│       └── yolopv2.pt
└── utils/
```

---

# ⚙️ Configuration

## Main Settings (`main.py`)

| Setting | Description | Default |
|--------|-------------|---------|
| CAPTURE_REGION | Game capture screen region | (0, 0, 1280, 720) |
| MONITOR_NUMBER | Target display for capture | 1 |
| PID (Kp, Ki, Kd) | Steering stability | 20, 0.001, 25 |
| CRUISE_THROTTLE | Normal throttle | 10000 |
| FULL_BRAKE | Emergency brake value | 32768 |

---

## Path Settings (`path_finder.py`)

Set your absolute installation path:

```python
YOLOP_DIR = r"D:\Path\To\Your\Project\Root"
```

---

# 🎮 Usage

### Run the complete system:

```
python path_finder.py
```

This launches:
- YOLOPv2 lane segmentation  
- YOLOv12 object detection  
- PID loop  
- vJoy joystick output  
- Real-time visualization  

---

# ⌨️ Controls & Hotkeys

| Key | Action |
|-----|--------|
| **T** | Pause/Resume |
| **Q** | Quit |
| **W** | Manual full throttle override |
| **S** | Manual brake override |

---

# 🧠 Technical Insights

## 1. Lane Detection Pipeline (YOLOPv2)
- Mask → Bottom-half sampling → Left/Right lane extraction  
- Compute mid-lane  
- Error = `image_center - lane_center`  
- EMA smoothes noise  
- PID computes steering force  

## 2. Object Detection Pipeline (YOLOv12)
- Track objects frame-to-frame  
- Compute bounding box area  
- Growth rate = (A_t - A_(t-1))  
- Threat score → decides:
  - Cruise Control  
  - Evasive Steering  
  - Emergency Brake  

## 3. vJoy Output Map
- All axes 0–32768  
- Steering neutral: **16384**  
- Throttle typically positive  
- Braking mapped to Rz axis  

---

# 🧩 Troubleshooting

| Issue | Fix |
|-------|------|
| `ImportError: utils.utils` | Ensure YOLOPv2 `utils/` folder is in project root |
| No steering | Check vJoy is installed + Device 1 enabled |
| No lane detection | Ensure `yolopv2.pt` exists at the correct path |
| Laggy performance | Lower resolution or use smaller YOLO model |
| Game not responding | Make sure GTA is set to “Gamepad Enabled” |

---

# 📜 Disclaimer

This system is meant **strictly for research and educational purposes**.  
Using automation in **online multiplayer** may violate game Terms of Service.

---

# ⭐ Credits
- YOLOPv2 by CAIC-AD  
- Ultralytics YOLO (YOLOv12)  
- GTA San Andreas (Rockstar Games)  
- vJoy (Virtual Joystick Driver)  

---

