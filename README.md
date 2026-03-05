# RobotProject
I will be creating a robot with a Raspberry pi, and just some wheels, a camera and thats all that connects to my computer and just follows orders, it extracts what I say with whisper or similar software and does it by moving motors using AI.

# Inspiration
This project is inspired by the Nvidia robot GR00T: https://github.com/NVIDIA/Isaac-GR00T. However that robot requires a lot of hardware, I want to create something similar, but simple, for my budget.

# Project Overview
Project: Modular Voice-Controlled Follower Robot
🎯 The Vision
This project aims to build a smart, responsive 4-wheeled robot capable of understanding natural language commands (like "Follow me," "Stop," or "Turn left") and executing them using real-time computer vision.

Instead of relying on expensive, heavy hardware or massive end-to-end foundation models on the robot itself, this project uses a distributed architecture:

The Brain (Local PC): Handles the heavy lifting. It uses a fast, LLM (gpt-oss-120b) to parse voice commands into actionable JSON data, and runs a lightweight vision model (YOLO11 Nano) to track targets visually.

The Body (Raspberry Pi): Rides on the robot chassis. It acts as the nervous system, streaming camera feed to the PC and listening over Wi-Fi for simple, direct motor commands to drive the wheels.

🧠 System Pipeline (The "Follow Me" Command)
Voice Input: User says, "Follow me."

LLM Parsing: The PC processes the audio and the LLM outputs a command like {"action": "follow", "target": "person"}.

Vision Activation: The PC triggers YOLO11 to find the "person" in the live video stream.

Tracking Logic: The PC calculates the bounding box coordinates of the person and computes steering/throttle values using a Proportional (P) controller.

Execution: The PC sends raw motor speeds to the Raspberry Pi over the local network to drive the robot toward the target.
