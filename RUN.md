# Hand Ninja — Run Instructions

Slice flying fruit with your real hand (Fruit-Ninja-style, MediaPipe + Pygame).

## Setup (one-time, in a virtual environment)

### Linux / macOS
```bash
cd hand_ninja
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Windows (PowerShell)
```powershell
cd hand_ninja
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

### Linux / macOS
```bash
source venv/bin/activate
python hand_ninja.py
```

### Windows
```powershell
.\venv\Scripts\Activate.ps1
python hand_ninja.py
```

## Controls
- **Index finger** → blade (move through fruit to slice)
- **SPACE** → restart after game over
- **ESC** → quit

## Notes
- Webcam required (auto-uses index 0). For best results, run in good lighting.
- 60 FPS target. Close other heavy apps if frame drops occur.
- On macOS, allow camera access for Terminal/IDE when prompted.
