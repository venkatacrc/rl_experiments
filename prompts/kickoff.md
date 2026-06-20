# Prompt for Claude Opus 4.7 (Extra High)

Copy everything below this line into a new conversation with Opus 4.7.

---

I'm planning a DIY project to build an autonomous vacuum robot for hard floors (tile + hardwood), and I want your help turning this into a concrete, technically rigorous build + learning plan. Please research current information where it would change your answer (parts prices/availability, Jetson Orin Nano power specs, current RL/robotics tooling) rather than relying only on training data.

## My background
- MS-level reinforcement learning coursework (Georgia Tech OMSCS) — comfortable with MDPs, dynamic programming, value/policy iteration, Q-learning/SARSA, policy gradients, and actor-critic methods (DQN, PPO, SAC, etc.)
- Strong Python; comfortable with PyTorch
- Comfortable going deep into math/algorithm details — don't oversimplify on my account

## Dual goal
1. **Learning**: a hands-on platform to implement and compare different RL algorithms on a real robotics coverage problem
2. **Product**: end up with something that actually vacuums/cleans my tile and hardwood floors reasonably well

Both matter, but if they trade off, prioritize the learning value — I'm fine with a robot that's mechanically simpler than commercial options as long as the RL pipeline is real and instructive.

## Hardware I already have
- NVIDIA Jetson Orin Nano 8GB Developer Kit
- 500GB NVMe SSD (for the Jetson)
- USB-C power supply (need you to verify whether this can adequately power the Jetson Orin Nano — see questions below)
- MacBook Pro, M1 chip (my dev machine — note: no CUDA, so on-device deep RL training isn't an option here)
- MSI GE75 Raider 10SE with NVIDIA GeForce RTX 2060 (6GB GDDR6) graphics card and an Intel HM470 chipset
- A USB camera
- Open to buying small/cheap additional components (motors, sensors, chassis parts, etc.)

## Budget
- ~$100 for additional parts (motors, motor driver, chassis, wheels, battery, vacuum mechanism, sensors, mounting hardware, etc.) — the Jetson/NVMe/camera/power supply above are already owned and not part of this budget
- Willing to stretch beyond $100 if there's a clear capability or learning payoff — please flag where stretch spending would meaningfully help vs. where it's not worth it

## Reference products (context only — not buying these, just calibrating expectations)
- PrettyCare C2-v2 Robot Vacuum and Mop Combo (~$100, 8000Pa suction, 5200mAh battery)
- A 2-in-1 robot vacuum/mop on Amazon (~under $100, 2000Pa suction, WiFi/app, self-charging, hard floor/low-pile carpet/pet hair)

I know my DIY build won't match these on raw suction power or polish — that's fine. Use them only as a rough sanity-check baseline for "does this DIY thing meaningfully clean," not as a target to beat on specs.

## Planned technical roadmap (please critique/adjust this)
- **Phase 0 — Hardware**: source and assemble a basic differential-drive chassis + vacuum mechanism, mount Jetson + camera, get a basic teleoperated/scripted control loop working
- **Phase 1 — Grid-world RL**: build a simple discretized simulation of a room and train a basic tabular policy (Q-learning/SARSA, or a simple policy gradient) for coverage path planning, to validate reward shaping for "coverage" and "cleanliness" before touching real hardware
- **Phase 2 — Camera-based RL**: train a small CNN policy (DQN/PPO/SAC or similar) using image observations from the USB camera, likely first in a simulator with domain randomization, then adapt toward the real robot
- **Phase 3 — Deploy**: quantize/distill the trained policy (e.g., export to ONNX, optimize with TensorRT, INT8 calibration, or distill into a smaller student network) so it runs in real time on the Orin Nano
- **Phase 4 — Evaluation**: measure coverage (% of floor area visited in a time budget) and cleanliness (some practical proxy, since I likely won't have a real dirt sensor) and iterate

## Specific things I want from you

1. **Feasibility & risks** — Is this whole plan realistic for the budget/hardware/implied scope? What are the biggest risks (mechanical, power, RL training, sim-to-real) and how would you de-risk them early?

2. **Bill of materials** — A parts list split into "$100 strict" and "reasonable stretch" tiers: drive motors + wheels, motor driver/H-bridge, chassis/frame, battery + charging circuit, vacuum mechanism (fan/brush/dustbin or a minimal version of one), bump/cliff/IR sensors, IMU, mounting for Jetson + camera, and power regulation from the robot's battery to the Jetson. Specific part suggestions and current prices/links where possible.

3. **Do I need LiDAR?** Analyze the tradeoffs of adding a budget LiDAR vs. camera-only / time-of-flight / ultrasonic / bump-sensor-only approaches, given that my core interest is a camera-based RL policy and my budget/compute (Orin Nano) constraints. Recommend whether to add it now, later, or skip it entirely, and what I'd do instead to compensate (e.g., visual odometry, lightweight SLAM, or a purely reactive policy).

4. **Power architecture** — Verify current specs on whether the Jetson Orin Nano 8GB Dev Kit can run reliably from USB-C power delivery for an untethered/mobile robot, what wattage/voltage it actually needs, and how I should separately power the drive motors and vacuum motor from a battery without interfering with the Jetson's power.

5. **Where/how to train** — Given a CUDA-less M1 MacBook Pro for dev and an Orin Nano onboard, where should RL training actually happen (cloud GPU, local CPU-only for phase 1, on-device for fine-tuning)? Recommend simulators: something lightweight/custom for the phase-1 grid-world, and an appropriate option (PyBullet, MuJoCo, Isaac Lab/Sim, Gazebo, or other) for the phase-2 camera-based policy, with trade-offs given my setup.

6. **RL algorithm choices per phase**, with justification — tabular methods for phase 1, and your recommended deep RL family/algorithm for the camera-based CNN policy in phase 2. Point me to relevant coverage-path-planning RL approaches/literature (reward shaping for coverage, exploration strategy, frame stacking, etc.) rather than generic Atari-style recipes.

7. **Sim-to-real transfer strategy** — domain randomization approach, how to fine-tune safely on the real robot, and how to bound real-world exploration so it doesn't damage itself or fall down stairs while learning.

8. **Quantization/distillation pipeline** — concrete recommended path to compress the trained CNN policy to run in real time on the Orin Nano (ONNX export, TensorRT, INT8 calibration, and/or distillation to a smaller student network), with a rough sense of the latency/FPS I should target for a usable control loop.

9. **Evaluation framework** — concrete methodology for measuring coverage % and a cleanliness proxy in both simulation and the real world (e.g., a flour/debris-style test similar to vacuum reviews), and how I might use the two reference commercial vacuums as a rough sanity check without expecting to match their raw cleaning specs.

10. **Concrete first sprint** — what should I actually start building and coding this week? Suggest a repo structure that will scale cleanly from phase 1 through phase 4.

## How to structure your answer
Please organize your response with headers matching the items above. Be specific and technical — equations, algorithm names, and concrete part numbers/links are all welcome. Where you're not certain about current pricing, availability, or hardware specs, search/verify rather than guessing, and flag explicit assumptions you're making.
