# **VLN Live Perception Pipeline — AP-VLM + Cola**

This project implements a **modular live perception pipeline** for Vision-Language Navigation (VLN) by combining two research papers:

- **AP-VLM** (Sripada et al., 2024): Active Perception Enabled by Vision-Language Models  
- **Cola** (Chen et al., NeurIPS 2023): Large Language Models are Visual Reasoning Coordinators

The pipeline takes a **live webcam feed** as input (swappable to ROS2 topic when a robot is available) and runs a full perception + coordination + active perception loop, producing a structured output report.

---

## **Project Overview**

The pipeline includes:

1. **Live Frame Acquisition** (`perception/frame_source.py`):
   - Abstraction layer over webcam / video file / ROS2 topic
   - Swap `source: webcam` → `source: ros_topic` in `config.yaml` for robot integration
   - No code changes needed anywhere else

2. **Modular Perception** (`perception/perception_module.py`):
   - **BLIP** (`Salesforce/blip-image-captioning-base`): scene captioning + 768-D visual embedding
   - **YOLOv8** (`yolov8n.pt`): object detection with direction and range labels
   - **MiDaS** (`DPT_Hybrid`): monocular depth estimation
   - Returns same schema as old database-based pipeline

3. **Cola Coordination** (`perception/cola_coordinator.py`):
   - Implements Cola prompt template (Table 1, NeurIPS 2023)
   - VLM-1 = BLIP caption, VLM-2 = YOLO object list
   - LLM (FLAN-T5 local or GPT-4o-mini) coordinates both for navigation-relevant scene understanding
   - Outputs: `scene_summary`, `relevant_objects`, `is_conclusive`, `confidence`, `action_hint`

4. **Active Perception Policy** (`perception/active_perception_policy.py`):
   - Implements AP-VLM iterative exploration loop
   - Uses `is_conclusive` flag from coordinator to decide: navigate or explore more
   - Maintains accumulated knowledge context (κ) across iterations
   - Terminates when confident or max iterations reached

5. **Report Generation** (`perception/report_generator.py`):
   - Per-frame perception log (caption, objects, depth, inference time)
   - Cola coordination log (scene summaries, confidence, action hints)
   - AP-VLM action trace (iteration-by-iteration decisions)
   - 4 matplotlib figures

---

## **Project Structure**

```
.
├── main.py                              # Main execution script
├── config.yaml                          # All pipeline settings
├── requirements.txt                     # Python dependencies
├── README.md                            # Project documentation
├── perception/
│   ├── __init__.py
│   ├── frame_source.py                  # Webcam / video / ROS2 input abstraction
│   ├── perception_module.py             # BLIP + YOLO + MiDaS per-frame inference
│   ├── cola_coordinator.py              # Cola dual-VLM LLM coordination
│   ├── active_perception_policy.py      # AP-VLM iterative exploration policy
│   └── report_generator.py             # results/ + figures/ output
├── notebooks/
│   └── demo_pipeline.ipynb             # Demo notebook (video file input)
├── results/                            # Generated TXT + JSON reports
└── figures/                            # Generated PNG plots
```

---

## **Installation**

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/vln-live-perception.git
cd vln-live-perception
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

**Python version**: 3.10.x (tested on Python 3.10.12)

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```
*downloads transformers too(heavy)*
> For GPU support: install PyTorch with CUDA from [pytorch.org](https://pytorch.org/get-started/locally/) before running pip install.

---

## **Execution Steps**

### 1. Run with Webcam (default)

```bash
python main.py
```

### 2. Run on Video File (no webcam needed)

```bash
python main.py --source video_file --video_path demo.mp4
```

### 3. Custom Navigation Instruction

```bash
python main.py --instruction "Go to the kitchen and find the table"
```

### 4. Perception Only (no LLM / coordinator)

```bash
python main.py --no_coordinator
```

### 5. Robot Integration (when robot is available)

Change one line in `config.yaml`:
```yaml
input:
  source: ros_topic          # was: webcam
  ros_topic: /camera/image_raw
```
Then: `pip install rclpy cv_bridge` and run normally. Everything else stays identical.

### 6. Jupyter Notebook Demo

```bash
jupyter notebook notebooks/demo_pipeline.ipynb
```

---

## **Configuration**

All settings are in `config.yaml`. Key options:

| Section | Key | Default | Description |
|---|---|---|---|
| `input` | `source` | `webcam` | `webcam` / `video_file` / `ros_topic` |
| `perception` | `device` | `cpu` | `cpu` or `cuda` |
| `coordinator` | `backend` | `hf` | `hf` (local FLAN-T5) or `openai` (GPT-4o-mini) |
| `active_perception` | `max_iterations` | `10` | AP-VLM max exploration steps |
| `pipeline` | `frame_interval` | `5` | process every N-th frame |

---

## **Results**

The pipeline generates:

### Text Report (`results/<session>_report.txt`)

```
[1] PERCEPTION SUMMARY (BLIP + YOLO + MiDaS)
    Mean inference time : 420.3 ms
    Mean objects/frame  : 3.20
    Top objects         : person(14), chair(9), tv(6), ...

[2] COLA COORDINATION SUMMARY (VLM-1 + VLM-2 → LLM)
    Mean confidence     : 0.712
    Conclusive frames   : 18 / 25

[3] AP-VLM ACTIVE PERCEPTION TRACE
    iter= 1 | frame=   5 | action=explore_more   | conf=0.45
    iter= 2 | frame=  10 | action=move_forward   | conf=0.71
    ...
    iter= 6 | frame=  30 | action=stop           | conf=0.82 [TERMINATED]
```

### Figures (`figures/`)

| File | Description |
|---|---|
| `*_object_detection_frequency.png` | Bar chart: most frequent YOLO detections |
| `*_inference_time_distribution.png` | Histogram: BLIP+YOLO+MiDaS inference time per frame |
| `*_confidence_over_frames.png` | Line plot: Cola coordinator confidence over time |
| `*_action_sequence.png` | AP-VLM action sequence across iterations |

### JSON Log (`results/<session>_full_log.json`)

Full structured log: per-frame perception + coordination + action data.

---

## **Key Functions**

| File | Description |
|---|---|
| `frame_source.py` | `build_source(cfg)` — factory for webcam / video / ROS2 |
| `perception_module.py` | `PerceptionModule.process_scene(image)` — BLIP + YOLO + MiDaS |
| `cola_coordinator.py` | `ColaCoordinator.coordinate(perception, instruction)` — dual-VLM LLM coordination |
| `active_perception_policy.py` | `ActivePerceptionPolicy.step(coord, instruction)` — AP-VLM policy step |
| `report_generator.py` | `ReportGenerator.generate()` — all outputs |
| `main.py` | `run(cfg)` — full pipeline orchestration |

---

## **Dependencies**

- Python 3.10+
- PyTorch ≥ 2.0
- transformers ≥ 4.37 (BLIP)
- ultralytics ≥ 8.0 (YOLOv8)
- timm (MiDaS dependency)
- opencv-python
- matplotlib
- PyYAML
- sentencepiece, accelerate (FLAN-T5)

Full list: `requirements.txt`

---

## **References**

1. Sripada, V., Carter, S., Guerin, F., & Ghalamzan, A. (2024). *AP-VLM: Active Perception Enabled by Vision-Language Models*. arXiv:2409.17641  
2. Chen, L., Li, B., Shen, S., Yang, J., Li, C., Keutzer, K., Darrell, T., & Liu, Z. (2023). *Large Language Models are Visual Reasoning Coordinators*. NeurIPS 2023. arXiv:2310.15166

---

## **Contributions**

Contributions are welcome! Feel free to open an issue or create a pull request.

---

## **License**

This project is licensed under the [MIT License](LICENSE).

---

## **Internship Context**

Developed as part of a Vision-Language Navigation (VLN) internship at **Sabudh Foundation**, supervised by **Prof. Sukhjit Singh Sehra**. This pipeline transitions the perception module from a pre-cached database setup to a modular live video feed architecture, ready for robot integration via ROS2.
