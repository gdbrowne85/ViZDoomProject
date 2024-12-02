CSCE 642 Project - Glen Browne - Carlos Meisel

Note: You will need a CUDA enabled GPU to run this project with good performance

## Description
This project uses VizDoom to train agents with deep reinforcement learning to complete a variety of tasks

## Prerequisites
- Ensure you have Python installed (version 3.6 or higher is recommended).
- Install `git` if it's not already available on your system.

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/gdbrowne85/ViZDoomProject
   cd ViZDoomProject
   ```

2. Install requirements.txt
```bash
   pip install -r requirements.txt
```
3. Run Individual Python Scripts as desired:

Attack Dodging:
```bash
python3 Dodge.py
```

Item Gathering:
```bash
python3 Gather_Items.py
```

High Accuracy Shooting:
```bash
python3 High_Accuracy.py
```

Main Task (Single DQN):
```bash
python3 Main_Task_Single_Network.py
```

Main Task (Multitask Transfer Learning with Selector Network):
```bash
python3 Main_Task_Multitask_Transfer.py
```


