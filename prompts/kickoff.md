I'm planning a DIY project to build an autonomous vacuum robot for hard floors (tile + hardwood), and I want this single response to serve as a complete, standalone project manual — covering everything from initial parts sourcing through a working, RL-controlled robot.

## Important: treat this as one-shot, no follow-up access
I will **not** be able to ask you follow-up questions or come back mid-project. This is my only interaction with you on this project, so your response needs to be self-sufficient enough that I can execute the entire project — start to finish — using only what you give me here, plus the external links you provide. Please:

- **Be exhaustive, not concise.** Err on the side of more detail, more code, more links, and more troubleshooting guidance, even if the response is very long. Length is not a constraint; incompleteness is the failure mode I'm worried about.
- **Include actual runnable starter code** for each phase (not just descriptions of what the code should do), with inline comments explaining the logic.
- **Include direct links for everything** I'd otherwise need to ask about: datasheets, product/purchase pages, library/API documentation, tutorials, reference repos or papers, and communities/forums where I can get unstuck (Jetson forums, relevant subreddits, Robotics Stack Exchange, GitHub issues, etc.) — since I can't ask you directly if something breaks.
- **Anticipate my natural follow-up questions** at each phase and answer them inline — common pitfalls, debugging tips, what to do if a part doesn't physically fit, what to do if training doesn't converge, what to do if sim-to-real transfer fails, etc.
- **Where there's a genuine judgment call** that depends on information you don't have (e.g., my specific floor plan, exact part tolerances), give a clear default recommendation **and** explain the alternative plus how I'd decide between them myself — don't leave it as an open question back to me.
- **Provide a table of contents** at the top of your response so I can navigate it as a reference document throughout the build.

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
- MacBook Pro, M1 chip (general dev machine — no CUDA)
- MSI GE75 Raider 10SE laptop, NVIDIA GeForce RTX 2060 (6GB GDDR6), Intel HM470 chipset — this is my CUDA-capable machine for local training
- Also able to spin up cloud GPU instances (A100/H100) for training if needed
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

## Deliverables I need in this single response

For each item below, include both the explanation/decision **and** any code, links, or reference material needed to actually execute it — assume I cannot come back and ask for the code or links separately.

1. **Feasibility & risks** — Is this whole plan realistic for the budget/hardware/implied scope? What are the biggest risks (mechanical, power, RL training, sim-to-real) and how would I de-risk them early? Include links to any cautionary write-ups from people who've attempted similar DIY robot builds.

2. **Bill of materials** — A parts list split into "$100 strict" and "reasonable stretch" tiers: drive motors + wheels, motor driver/H-bridge, chassis/frame, battery + charging circuit, vacuum mechanism (fan/brush/dustbin or a minimal version of one), bump/cliff/IR sensors, IMU, mounting for Jetson + camera, and power regulation from the robot's battery to the Jetson. For every part: a direct purchase link (Amazon/Adafruit/SparkFun/etc.), current approximate price, and a link to its datasheet or wiring guide. Also include a wiring diagram description (or ASCII/text diagram) showing how everything connects to the Jetson's GPIO/USB/power.

3. **Do I need LiDAR?** Analyze the tradeoffs of adding a budget LiDAR vs. camera-only / time-of-flight / ultrasonic / bump-sensor-only approaches, given that my core interest is a camera-based RL policy and my budget/compute (Orin Nano) constraints. Recommend whether to add it now, later, or skip it entirely, and what I'd do instead to compensate (e.g., visual odometry, lightweight SLAM, or a purely reactive policy). If you recommend a specific LiDAR unit, include its purchase link, datasheet, and a link to a Jetson driver/integration guide.

4. **Power architecture** — Verify current specs on whether the Jetson Orin Nano 8GB Dev Kit can run reliably from USB-C power delivery for an untethered/mobile robot, what wattage/voltage it actually needs (link the official spec sheet), and how I should separately power the drive motors and vacuum motor from a battery without interfering with the Jetson's power. Include a link to any relevant NVIDIA forum threads on USB-C-powered mobile Jetson builds if power delivery is a known pain point.

5. **Where/how to train** — I have a local CUDA GPU (RTX 2060, 6GB GDDR6) on the MSI GE75 Raider, a CUDA-less M1 MacBook Pro for general dev, an Orin Nano onboard the robot, and the ability to spin up cloud GPU instances (A100/H100) when needed. Give me a clear decision rule: what's realistic to train locally on the 6GB RTX 2060 (e.g., phase-1 grid-world, small-scale phase-2 prototyping) vs. when it's actually worth paying for cloud A100/H100 time (e.g., larger CNN policies, massively parallel environment rollouts, domain-randomized sim-to-real training) — including a rough sense of cost/time tradeoffs and which cloud providers/services are easiest to get started with (links to their docs/pricing). Recommend simulators: something lightweight/custom for the phase-1 grid-world (fine on CPU), and an appropriate option (PyBullet, MuJoCo, Isaac Lab/Sim, Gazebo, or other) for the phase-2 camera-based policy — including install/setup links and whether Isaac Sim/Lab makes sense locally on the 6GB card vs. needing cloud GPUs to run well.

6. **RL algorithm choices per phase**, with justification — tabular methods for phase 1, and your recommended deep RL family/algorithm for the camera-based CNN policy in phase 2. Point me to relevant coverage-path-planning RL papers/repos (reward shaping for coverage, exploration strategy, frame stacking, etc.) with links, rather than generic Atari-style recipes. Include a runnable code skeleton for the phase-1 tabular agent and the phase-2 deep RL agent (e.g., using Gymnasium + Stable-Baselines3, or a custom training loop if more appropriate — your call, but justify it).

7. **Sim-to-real transfer strategy** — domain randomization approach, how to fine-tune safely on the real robot, and how to bound real-world exploration so it doesn't damage itself or fall down stairs while learning. Link any reference implementations or papers on sim-to-real for small ground robots.

8. **Quantization/distillation pipeline** — concrete, step-by-step path to compress the trained CNN policy to run in real time on the Orin Nano (ONNX export, TensorRT, INT8 calibration, and/or distillation to a smaller student network), with links to the relevant NVIDIA TensorRT/JetPack documentation and example repos, plus a rough sense of the latency/FPS I should target for a usable control loop.

9. **Evaluation framework** — concrete methodology for measuring coverage % and a cleanliness proxy in both simulation and the real world (e.g., a flour/debris-style test similar to vacuum reviews), and how I might use the two reference commercial vacuums as a rough sanity check without expecting to match their raw cleaning specs. Include a code skeleton or logging approach for tracking these metrics across training runs.

10. **Concrete build sequence** — a week-by-week or milestone-by-milestone execution plan from "parts arrive" to "robot cleans autonomously," covering hardware assembly, software setup, and RL training in the right order, with a suggested repo structure that scales cleanly from phase 1 through phase 4. Include the exact first commands/scripts I should run on day one (Jetson OS flashing/setup links, environment setup, etc.).

## Resource & reference appendix
At the end of your response, consolidate **every link you've referenced** into one organized appendix, grouped under: Datasheets & hardware docs / Purchase links / Software & library docs / Simulators / Papers & further reading / Communities & troubleshooting forums. This is the section I'll return to whenever I get stuck.

## Troubleshooting & FAQ
Add a final section anticipating the issues I'm most likely to hit at each phase (hardware assembly, power, RL training not converging, sim-to-real gap, quantization/deployment errors) with concrete fixes or diagnostic steps for each — written as if answering the follow-up questions I won't get the chance to ask.

## How to structure your answer
Use the table of contents and headers matching the sections above. Be specific and technical — equations, algorithm names, code blocks, and concrete part numbers/links are all expected, not optional extras. Where you're not certain about current pricing, availability, or hardware specs, search/verify rather than guessing, and clearly flag any assumptions you're making.
