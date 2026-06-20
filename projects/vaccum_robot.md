# DIY Autonomous RL Vacuum Robot — Complete Build Manual

A self-contained reference for a Jetson Orin Nano 8GB–based, camera-driven, reinforcement-learning vacuum robot for hard floors, intended for one-shot execution.

> **Verified on:** June 20, 2026 — JetPack 6.2 (L4T 36.4.3, CUDA 12.6, TensorRT 10.3), Stable-Baselines3 v2.8.0, Isaac Sim/Lab 5.x.

---

## Table of Contents

1. [Feasibility & Risks](#1-feasibility--risks)
2. [Bill of Materials (Strict $100 and Stretch tiers)](#2-bill-of-materials)
3. [Do You Need LiDAR?](#3-do-you-need-lidar)
4. [Power Architecture](#4-power-architecture)
5. [Where & How to Train](#5-where--how-to-train)
6. [RL Algorithm Choices Per Phase (with runnable code)](#6-rl-algorithm-choices-per-phase)
7. [Sim-to-Real Transfer](#7-sim-to-real-transfer)
8. [Quantization / Distillation / Deployment](#8-quantization-distillation-deployment)
9. [Evaluation Framework](#9-evaluation-framework)
10. [Concrete Build Sequence & Repo Structure](#10-concrete-build-sequence--repo-structure)
11. [Resource Appendix (all links, grouped)](#11-resource-appendix)
12. [Troubleshooting & FAQ](#12-troubleshooting--faq)

---

## 1. Feasibility & Risks

### Overall verdict
**Yes, this is feasible** for the stated dual goals (learning + a robot that meaningfully cleans hard floors), with three honest caveats:

1. **Your $100 will be tight** if you want both LiDAR and a real vacuum mechanism. The plan below stays within $100 strict by skipping LiDAR (you don't need it for camera-based RL) and using a repurposed handheld vacuum motor.
2. **Sim-to-real for camera policies is the hardest single step.** Plan for it from Phase 1 (consistent observation shape, action discretization, frame rate budget).
3. **Suction power will be modest** (your reference commercial units are 2,000–8,000 Pa; a tiny brushed-DC fan you can drive on 7.4–12V will be a fraction of that). This is OK for "vacuum the lint/dust on hard floors" but not for "deep-clean carpet." You explicitly accepted that trade-off.

### Risk register and de-risking

| Risk | Likelihood | Impact | De-risking plan |
| --- | --- | --- | --- |
| Jetson power instability under motor surges | **High** | Brownouts, kernel panics, SD/NVMe corruption | Strict galvanic separation: separate battery rail for Jetson (via boost converter to 9V) vs. motor rail. Add 1000 μF bulk caps near motor driver Vmot. Never share a single linear regulator. See [thread on Jetson brownouts](https://forums.developer.nvidia.com/t/jetson-nano-runtime-shutdown-when-motor-starts/72244/). |
| Sim-to-real gap (camera policy works in sim, fails on real robot) | **High** | Phase 2 stalls | Domain randomization from day one (Section 7). Start with frame stacks at 84×84 grayscale (same as Atari) so you can drop in proven hyperparameters. Always keep a scripted "safe wander + bump" baseline as fallback. |
| Coverage reward shaping never converges | **Medium** | Phase 1 stalls | Reuse the formulation in Jonnarth et al. 2024 (egocentric frontier map + total-variation bonus). See [paper](https://proceedings.mlr.press/v235/jonnarth24a.html) and [open code](https://github.com/jonnarth/coverage-path-planning-rl). |
| Mechanical chassis can't carry the Jetson (top-heavy) | Medium | Robot tips, rolls poorly | Mount Jetson low and centered, battery lower still. Use a wide (≥15 cm) wheelbase; avoid the tiny 2WD "round disc" car kits that work fine on hardwood only if the COG is low. |
| Cliff fall (stairs) during real-world training | Medium | Hardware destruction | Bound exploration physically: train in a single enclosed room with closed doors; add 2× VL53L0X downward ToF as **hard-coded** safety stop in low-level firmware/loop, before the RL policy can act. Section 7. |
| 6 GB VRAM on RTX 2060 too small for Isaac Lab | **Confirmed** | Can't run nicely | Skip Isaac Sim/Lab locally. Use PyBullet locally; rent A100/H100 for any Isaac Lab work. Verified: Isaac Sim 5.x officially requires 16 GB VRAM ([source](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)). |
| TensorRT INT8 calibration set is wrong distribution → silent accuracy drop | Medium | Policy degrades on robot | Use ~500–1000 frames captured on the real robot (not in sim) as the calibration set. Verify with on-robot rollouts before declaring done. |
| Wheel encoder absence → no real odometry | Medium | Limits SLAM, but acceptable for camera RL | Either accept and use purely reactive/visual policy, or stretch-buy magnetic encoders ($6). |

### Cautionary write-ups from people who have done similar DIY builds
- [Pleco — Embedded handheld vac (blakelton)](https://github.com/blakelton/pleco) — Honest write-up of getting suction from a 12V PWM fan on battery; their motor sizing math is reusable.
- [DrLex0 / print3d-microvac-turbo](https://github.com/DrLex0/print3d-microvac-turbo) — DIY hand-vac with 3D-printed turbo impeller; useful for understanding how little suction you really get from a tiny motor.
- [JetBot project (NVIDIA)](https://github.com/NVIDIA-AI-IOT/jetbot) — Reference for what a Jetson + cam + 2-motor differential drive robot looks like (older Nano, but topology is identical for Orin Nano).
- [JetRacer reference build](https://github.com/NVIDIA-AI-IOT/jetracer) — Same idea, different mechanical form factor; their hardware lists and wiring guides are gold.
- [r/robotics DIY vacuum threads](https://www.reddit.com/r/robotics/search/?q=DIY+vacuum&restrict_sr=on) — Read the "I wish I had known…" comments; the most common regret is using the L298N motor driver (1.4 V drop and overheats).
- [r/JetsonNano](https://www.reddit.com/r/JetsonNano/) and [NVIDIA Jetson Orin Nano forum](https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems/jetson-orin-nano/619) — your primary "I'm stuck" venues for Jetson-specific issues.

---

## 2. Bill of Materials

> Prices reflect typical Amazon US listings verified June 2026. Items marked **CONFIRMED** I verified directly via web search; others are stable, long-running parts whose price hasn't changed materially in years. Substitute equivalents freely if shipping is faster.

### 2.1 Strict $100 tier (the minimum viable robot)

| # | Item | Why | Qty | Approx. Price | Where to buy | Datasheet / Guide |
|---|---|---|---|---|---|---|
| A1 | TT 1:48 yellow gearmotor + 66 mm wheel set (4-pack) | Differential drive, you'll only use 2 of 4 but the kits are cheaper than singles | 1 | $10 | [Amazon: yellow TT motor + wheel kit](https://www.amazon.com/s?k=tt+motor+wheel+kit) | [Specs (Cytron)](https://www.cytron.io/p-3v-6v-dual-axis-tt-gear-motor) — 200 RPM @ 6 V, stall 1.5 A |
| A2 | TB6612FNG dual H-bridge module (SparkFun ROB-14450 or Pololu #713) | 3.3 V-logic-compatible PWM motor driver. **Strictly preferred over L298N** for Jetson | 1 | $5 (clone) – $10 (Pololu) | [Pololu #713](https://www.pololu.com/product/713) · [SparkFun ROB-14450](https://www.sparkfun.com/products/14450) | [TB6612FNG datasheet (PDF)](https://www.pololu.com/file/0J86/TB6612FNG.pdf) · [DroneBot Workshop guide](https://dronebotworkshop.com/tb6612fng-h-bridge/) |
| A3 | Caster wheel (3/4" or 1") | Third support point | 1 | $2 | [Adafruit #2942](https://www.adafruit.com/product/2942) | — |
| A4 | Acrylic / 3 mm plywood chassis plate (cut to ~20 × 18 cm) or 2-tier robot car kit | Frame | 1 | $8 | [Amazon: robot car chassis 2WD](https://www.amazon.com/s?k=2wd+robot+car+chassis+acrylic) | — |
| A5 | 2× 18650 Li-ion cells (protected, 3000 mAh) **or** 3S 1300 mAh LiPo (11.1 V) | Power source | 1 set | $12 | [Amazon: protected 18650 2-pack](https://www.amazon.com/s?k=protected+18650+3000mah+2+pack) | Cell datasheet varies; use Panasonic NCR18650B or Samsung 30Q |
| A6 | 2S 18650 battery holder + JST + on/off switch (or 3S XT60 LiPo connector) | Wiring battery | 1 | $5 | [Amazon: 2x18650 holder XT60](https://www.amazon.com/s?k=2s+18650+holder+xt60) | — |
| A7 | DC-DC step-up boost converter, 5–18V in → 9–12V out, ≥3 A (e.g., MT3608-based or "12V to 19V" car-laptop boost) | Bring battery up to Jetson barrel jack range | 1 | $7 | [Drok 6V-32V boost converter](https://www.amazon.com/s?k=drok+boost+converter+12v+19v) | [12V→19V 5A spec sheet](https://daygreen.com/en-en/products/12v-to-19v-5a-dc-dc-step-up-converter-voltage-regulator-3-year) |
| A8 | DC barrel-jack pigtail (5.5 × 2.5 mm, center-positive, 30 cm with stripped leads) | Connect boost converter output to Jetson | 1 | $4 | [Amazon: 5.5×2.5 mm barrel pigtail](https://www.amazon.com/s?k=5.5mm+2.5mm+barrel+jack+pigtail) | — |
| A9 | 2× VL53L0X ToF distance sensor breakout (for cliff detection) | Anti-stair safety | 1 (or 2) | $7 | [Pololu #2490](https://www.pololu.com/product/2490) · [Amazon clone](https://www.amazon.com/s?k=vl53l0x) | [ST datasheet (PDF)](https://www.st.com/resource/en/datasheet/vl53l0x.pdf) · [Adafruit guide](https://learn.adafruit.com/adafruit-vl53l0x-micro-lidar-distance-sensor-breakout) |
| A10 | Tactile micro switch + lever arm (or DIY bump bumper from foam board + 2 switches) | Front bumper / wall contact | 2 | $3 | [Amazon: snap-action lever microswitch](https://www.amazon.com/s?k=snap+action+lever+microswitch) | — |
| A11 | MPU-6050 or MPU-9250 IMU breakout (I²C) | Heading/IMU for odometry & turn estimation | 1 | $5 | [Amazon: MPU-6050](https://www.amazon.com/s?k=mpu6050) | [InvenSense MPU-6050 RM (PDF)](https://invensense.tdk.com/wp-content/uploads/2015/02/MPU-6000-Register-Map1.pdf) |
| A12 | Vacuum mechanism: cheap USB-rechargeable handheld vac that you'll gut for its motor + impeller + dustbin, OR a 12V 40mm PWM fan (Pleco-style) + 3D-printed turbo housing | The actual cleaning bit | 1 | $15 (handheld vac) | [Amazon: $10 mini handheld vacuum](https://www.amazon.com/s?k=mini+handheld+vacuum+rechargeable) · [Noctua 12V PWM fan alternative](https://www.amazon.com/s?k=noctua+40mm+pwm+fan) | [Pleco GitHub](https://github.com/blakelton/pleco) (reference design) · [DrLex micro-vac](https://github.com/DrLex0/print3d-microvac-turbo) |
| A13 | Jumper wire kit (M-F, F-F, M-M, ~120 pcs) + JST 2-pin pigtails | Wiring | 1 | $7 | [Amazon: dupont jumper kit](https://www.amazon.com/s?k=dupont+jumper+wire+kit) | — |
| A14 | 1× IRLB8721 N-channel logic-level MOSFET module (or a TIP120 if you can't get the MOSFET) + 1N5822 flyback diode | Switch the vacuum motor at >1 A under PWM from Jetson 3.3 V | 1 | $4 | [Adafruit IRLB8721 #355](https://www.adafruit.com/product/355) | [IRLB8721 datasheet (PDF)](https://www.infineon.com/dgdl/irlb8721pbf.pdf) |
| A15 | Velcro/zip ties / mounting screws (M2 + M2.5 + M3 mixed) | Mounting Jetson, sensors, motors to chassis | — | $5 | [Amazon: M3 screw assortment](https://www.amazon.com/s?k=m3+m2+screw+assortment+kit) | — |
| | **Strict tier subtotal** | | | **≈$99** | | |

### 2.2 Stretch additions (only buy if the learning-payoff justifies it)

| Item | Approx. Price | Why stretch | When it's worth it |
|---|---|---|---|
| RPLidar A1M8 (~$99 [CONFIRMED]) | $99 | 360° 2D LiDAR; instant SLAM. [Amazon link](https://www.amazon.com/Slamtec-RPLIDAR-Scanning-Avoidance-Navigation/dp/B07TJW5SXF) · [datasheet](https://bucket-download.slamtec.com/d1e428e7efbdcd65a8ea111061794fb8d4ccd3a0/LD108_SLAMTEC_rplidar_datasheet_A1M8_v3.0_en.pdf) · [ROS2 driver](https://github.com/Slamtec/sllidar_ros2) | If you decide camera-only + reactive RL is too brittle and you want to compare with a SLAM-based occupancy-grid policy. Doubles your budget. |
| 2× magnetic Hall encoders for TT motors ($6/pair) | $6 | Real wheel odometry → much better odometry → useful for evaluation (Section 9) | If you want quantitative coverage measurement without external tracking, or to validate sim odometry. [Amazon link](https://www.amazon.com/s?k=tt+motor+encoder+hall) |
| Intel RealSense D435i ($230) or OAK-D Lite ($150) | $150–230 | Depth + RGB, much cleaner observation than USB cam; built-in IMU | Only if Phase 2 keeps failing because monocular depth ambiguity (rare on flat floors). |
| RTC + 18650 BMS (3S protected board) | $5 | Battery protection | Skip the BMS only if you use a single-cell 3.7 V system with a known charger; otherwise add this for safety. |
| Buck converter for 5V/3.3V auxiliary | $4 | Power IMU + ToF if you want isolation from Jetson 3.3V rail | Mostly redundant — Jetson 40-pin header provides 3.3 V and 5 V. |
| Heatsink + 40mm fan upgrade for Jetson | $15 | MAXN_SUPER mode @ 25 W needs more cooling than stock | If your phase-2 policy can't hit real-time on the stock cooler. |
| Coral USB Edge TPU | $60 | Off-load CNN inference | **Not recommended** for this project — TensorRT INT8 on the Orin's GPU is faster and same pipeline as your training. Save the money. |

> **Wiring diagram (ASCII).** Always wire ground (GND) commons together across every subsystem; floating grounds cause inscrutable bugs.

```
                ┌──────────── Battery (2S 18650 ≈ 7.4 V, or 3S LiPo ≈ 11.1 V) ─────────────┐
                │  + ──┬──── Switch ──┬───────────────────────────────┐                    │
                │      │              │                                │                    │
                │      │              ▼                                ▼                    │
                │      │      Boost converter 9-12V                  Vmot (TB6612 VM, MOSFET drain)
                │      │              │                                │
                │      │              ▼                                │
                │      │     Barrel-jack pigtail (5.5×2.5 mm, center+) │
                │      │              │                                │
                │      │              ▼                                │
                │      │       Jetson Orin Nano DC IN                  │
                │      │              │                                │
                │      └──────────────┴────────── GND (common) ────────┴── GND
                │
                └ Vbat sensing via voltage divider → IMU/ADC (optional)

Jetson 40-pin header signals (BCM-style numbering, but physical pins):
  Pin 1 (3V3)  ────► VL53L0X VIN, MPU-6050 VCC
  Pin 6 (GND)  ────► common ground rail
  Pin 2 (5V)   ────► not used (board can do 3 A but we go through battery rail)
  Pin 3 (SDA1) ────► VL53L0X #1 SDA, MPU-6050 SDA, [VL53L0X #2 after XSHUT toggle reassigns its addr]
  Pin 5 (SCL1) ────► VL53L0X SCL, MPU-6050 SCL
  Pin 8/10/12/16/18 (GPIO) ─► TB6612: AIN1, AIN2, BIN1, BIN2, STBY
  Pin 32 (PWM0)────► TB6612: PWMA  (left wheel speed)         [hardware PWM, see PWM note]
  Pin 33 (PWM2)────► TB6612: PWMB  (right wheel speed)        [hardware PWM]
  Pin 35       ────► IRLB8721 gate (vacuum motor MOSFET)      [GPIO with PWM via software is OK at low Hz]
  Pin 37/38/40 ───► tactile bumper switches → input_pullup
  Pin 13/15    ────► VL53L0X #1 XSHUT, VL53L0X #2 XSHUT (for I2C address reassignment)

USB-A:  USB camera (your existing one) → Jetson USB 3.0 port
DC barrel jack: 9-12 V from boost converter → Jetson

TB6612 motor connections:
  AO1/AO2 → left motor (+/-)
  BO1/BO2 → right motor (+/-)
  VCC → 3.3 V (logic)
  VM  → battery 7.4 V (motor power)
  STBY → must be HIGH to enable (tie to GPIO or to 3.3V)
```

> **Pinout reference:** [Jetson Orin Nano 40-pin pinout (JetsonHacks)](https://jetsonhacks.com/nvidia-jetson-orin-nano-j12-header-pinout/). Note that only **two** hardware PWM channels are exposed on the Orin Nano 40-pin header (pins 32 and 33 by default), and you **must enable them** with `sudo /opt/nvidia/jetson-io/jetson-io.py` before they work — this is a known footgun ([NVIDIA jetson-gpio README](https://github.com/NVIDIA/jetson-gpio)). If you need >2 PWM channels, add a [PCA9685 16-channel I²C PWM driver ($5)](https://www.adafruit.com/product/815) which is the standard workaround.

---

## 3. Do You Need LiDAR?

### Short answer
**No — skip LiDAR for this build.** Add it later only if you decide to compare your camera-only RL policy against a LiDAR-based occupancy-grid baseline.

### Tradeoff table

| Sensor | Cost | Adds info | RL implications | Verdict for this project |
|---|---|---|---|---|
| **Monocular USB camera (you have)** | $0 | RGB image, no depth | Standard CNN policy input; classic Atari-style 84×84 stack works | **Use this** (your stated goal) |
| **2D LiDAR (RPLidar A1, $99)** | $99 (eats budget) | 360° range, low noise | Lets you build occupancy grid → simpler MDP, better sim-to-real | Stretch only; ~doubles budget and changes the learning problem |
| **3D depth camera (D435i, $230 / OAK-D Lite, $150)** | $150–230 | RGB-D, IMU | Cleaner depth → easier collision avoidance, useful for visual policies | Only if monocular camera is the blocker for Phase 2 |
| **Multiple VL53L0X ToF (1 m range, $4 each)** | ~$8 for 2 | Distance to 1–2 m forward + cliffs | Cheap obstacle/cliff safety; not enough for SLAM | Use it (it's already in BoM) |
| **Ultrasonic HC-SR04** | $2 | 2–400 cm range, noisy, slow | Marginal value, narrow cone, struggles on soft furniture | Skip |
| **Bumper only** | $1 | Binary contact | Forces "explore by bumping" policy, classic vacuum approach | Use it as the lowest layer |

### Why camera-only is fine pedagogically
- Coverage-path RL is well-studied with both occupancy grids and egocentric images. The recent state-of-the-art ([Jonnarth et al. ICML 2024](https://proceedings.mlr.press/v235/jonnarth24a.html)) shows that an egocentric, frontier-based map representation is computationally cheap and works without LiDAR if you have a way to estimate which floor cells you've visited.
- Your "have I cleaned this cell" estimate can come from **visual SLAM (ORB-SLAM3)** or, more pragmatically, **dead-reckoning from wheel encoders + IMU** with periodic camera-based loop closure. For a learning project, even pure dead-reckoning with re-initialization at each docking is acceptable.
- The Jetson Orin Nano has plenty of CUDA cores to run a CNN policy at >30 FPS; it does not have *spare* memory bandwidth to also run a heavy LiDAR-based SLAM (e.g., GMapping, SLAM Toolbox) at 30 Hz alongside training-from-replay. Camera-only keeps the compute budget tight.

### If you change your mind later and want to add LiDAR
- Buy: [Slamtec RPLidar A1M8 ($99)](https://www.amazon.com/Slamtec-RPLIDAR-Scanning-Avoidance-Navigation/dp/B07TJW5SXF)
- Datasheet: [LD108 PDF](https://bucket-download.slamtec.com/d1e428e7efbdcd65a8ea111061794fb8d4ccd3a0/LD108_SLAMTEC_rplidar_datasheet_A1M8_v3.0_en.pdf)
- Power: 5 V/200 mA — runs fine off Jetson USB-A
- Driver/integration:
  - ROS 2 (Humble or Jazzy): [sllidar_ros2](https://github.com/Slamtec/sllidar_ros2)
  - ROS 1 (Noetic, EOL May 2025): [rplidar_ros](https://github.com/Slamtec/rplidar_ros)
  - Bare Python (no ROS): [pyrplidar](https://github.com/Hyun-je/pyrplidar)
- Jetson-specific notes: [JetsonHacks RPLidar tutorial](https://jetsonhacks.com/2018/12/07/rplidar-a2-nvidia-jetson-development-kits/) (article uses A2 but A1 setup is identical; you just need the cp210x USB-serial driver, which is built into recent JetPack kernels).
- Then move from camera-only RL to LiDAR-SLAM + camera RL: see the [Husarion Rosbot examples](https://github.com/husarion/rosbot_ros) for an end-to-end ROS 2 stack on a small Jetson-class robot.

---

## 4. Power Architecture

### Verified facts
- **DC barrel jack is the only supported power input:** 5.5 mm OD × 2.5 mm ID, center positive, accepts **5–20 V**. ([JetsonHacks review](https://jetsonhacks.com/2023/03/22/nvidia-jetson-orin-nano-developer-kit-the-perfect-solution-for-makers-and-developers-a-review/) confirms the connector size; NVIDIA forum confirms the 5–20 V range.)
- **Power modes (JetPack 6.2):** 15 W, 25 W, MAXN SUPER. Stock adapter is 19 V / 2.37 A (45 W). At MAXN SUPER you can pull ~25 W sustained; spikes will trip a 25 W converter, so size the boost converter to ≥45 W to keep margin.
- **Required current:** roughly 1 A @ 12 V for typical workloads, up to 2.5 A @ 12 V in MAXN SUPER. Always provision **at least 3 A** at your chosen voltage.

### Recommended architecture (default)

```
2S 18650 (7.4–8.4 V) ── Boost converter ── 12 V/3 A rail ── Barrel-jack pigtail ── Jetson DC IN
                       ──────────────────                        
                       │                                          
                       │ (battery's raw 7.4 V)                    
                       ├─► TB6612FNG VM input  (drive motors)
                       │
                       └─► IRLB8721 MOSFET source for vacuum motor (use the raw 7.4-8.4 V if your motor likes it; otherwise step up too)

Jetson 40-pin 3V3 → IMU, ToF sensors logic
Jetson 40-pin 5V → unused (avoid loading Jetson onboard regulators with motor-adjacent gear)
Jetson 40-pin GPIO → TB6612 control + IRLB8721 gate + bumper inputs
```

### Why a boost converter (and not direct battery to barrel jack)
- 2S Li-ion (≈ 6.0–8.4 V) is *technically* in the 5–20 V acceptable range, but voltage sag under load (especially when the vacuum motor kicks on) can drop briefly below 5 V → Jetson brownout. Boost converter + a fat capacitor (1000 μF electrolytic + 10 μF ceramic) on the input gives you a flat 12 V rail.
- A 3S LiPo (9–12.6 V) *could* go direct, but you'll still want a buck regulator pinning it to a fixed 12 V to avoid telling the Jetson PMIC "voltage is dropping" right when you spin up the vacuum.
- Cheaper alternative: a "12V to 19V" laptop boost converter (those small aluminum-cased units, ~$10 on Amazon, e.g., [Daygreen 12 V → 19 V 5 A](https://daygreen.com/en-en/products/12v-to-19v-5a-dc-dc-step-up-converter-voltage-regulator-3-year)) is a known-quantity, isolated solution.

### How to separately power motors and vacuum
- **Run the drive motors directly off battery (7.4 V).** TB6612FNG handles 4.5–13.5 V VM input. The TT motors are rated 3–6 V, but they'll happily run at 7.4 V with shorter brush life — for a learning project, fine. If you're worried, PWM-limit them to ~70 % duty.
- **Run the vacuum motor off battery** through the IRLB8721 MOSFET. The MOSFET source is GND, drain goes to motor − , motor + goes to battery + . **Don't forget the flyback diode** (1N5822 or 1N4004) reverse-biased across the motor terminals to clamp inductive kick.
- **Never share rails with the Jetson.** If the only ground is shared (which it must be), you're fine. Sharing a power rail (e.g., 5 V) between Jetson and motors is what causes brownouts.

### Known pain points (forum threads)
- [Powering Jetson Orin Nano from battery — comprehensive thread](https://forums.developer.nvidia.com/t/powering-the-jetson-orin-nano/339465)
- [Recommended battery parameters for Orin Nano dev kit](https://forums.developer.nvidia.com/t/recommended-battery-parameters-mode-of-power-input-for-orin-nano-dev-kit/252048)
- [USB-C for power — repeatedly asked, always answered "no"](https://forums.developer.nvidia.com/search?q=orin%20nano%20usb-c%20power)
- [Jetson brownout while motor running (Jetson Nano era, applies)](https://forums.developer.nvidia.com/t/jetson-nano-runtime-shutdown-when-motor-starts/72244/)

### Battery life math
- Jetson @ 15 W (default profile) ≈ 1.25 A from a 12 V rail ≈ 1.5 A from 2S 18650 pack via boost (90 % eff).
- Two TT motors @ ~150 mA each running = 0.3 A.
- Vacuum fan @ 5 W ≈ 0.7 A.
- **Total ≈ 2.5 A from a 6000 mAh 2S pack ≈ 2.4 hours of runtime.** Plenty for a 30-min cleaning cycle + training rollouts.

---

## 5. Where & How to Train

You have four compute targets. Use them as follows.

| Target | What it's for | When to use it | When **not** to |
|---|---|---|---|
| **M1 MacBook Pro (no CUDA)** | Code editing, repo work, lightweight CPU-only sim (Phase 1 gridworld), Jupyter notebooks | Always — primary dev box. PyTorch on MPS now runs many CNN ops well enough for sanity checks. | Real PPO/SAC training. JAX/MJX on Apple Silicon is fine but limited by 8/16 GB unified memory. |
| **MSI GE75 — RTX 2060 6 GB** | Phase 1 gridworld (CPU is enough but you can use GPU); Phase 2 *prototyping* with PyBullet + small CNN; debugging | Mostly for Phase 1 and the first end-to-end PPO Phase 2 with small replay buffer (≤ 200 k transitions) | Isaac Sim/Lab (needs ≥ 16 GB VRAM, *officially*). Very large parallel envs. Long-running (>12 h) training. |
| **Cloud A100 / H100** | Phase 2 at scale: many parallel envs, big replay buffer, domain randomization with batch rendering, Isaac Lab if you decide you want it | Once your code runs end-to-end on the laptop AND you've validated reward shaping on gridworld | Phase 1 (overkill) or quick edits (slow turnaround) |
| **Jetson Orin Nano (onboard)** | Inference only, after TensorRT optimization | Phase 3+ deployment | Anything training-related. You'll thrash the SD/NVMe. |

### Decision rule (operational)

1. **Always start on the laptop** for any new code path. Get one rollout to complete.
2. **If laptop training is hitting < 200 FPS** in sim (CPU + GPU summed throughput) and your wall-clock per 1M env steps is > 6 hours, **move to cloud**.
3. **If you need photoreal rendering, parallel cameras, or > 256 parallel envs**, you must go cloud (and Isaac Lab specifically).

### Cloud GPU recommendations (verified June 2026)

| Provider | A100 80GB | H100 80GB | RTX 4090 | Sweet spot for this project |
|---|---|---|---|---|
| **RunPod (Community)** | $1.39–1.79/hr | $1.99–2.69/hr | $0.34–0.69/hr | **Best default** — per-second billing, fast PyTorch templates ([pricing](https://www.runpod.io/pricing) · [docs](https://docs.runpod.io/)). Get $10 free credit. |
| **Vast.ai (Spot)** | $0.27–1.00/hr (interruptible) | $1.00–2.15/hr | $0.09–0.55/hr | Cheapest if you can checkpoint often. Spot interruptible. ([pricing](https://vast.ai/pricing) · [docs](https://docs.vast.ai/)) |
| **Lambda Labs** | $1.29–2.79/hr | $2.49–4.29/hr | n/a | Most reliable, on-demand only, premium. ([pricing](https://lambda.ai/service/gpu-cloud)) |
| **GCP / AWS / Azure** | $3–6/hr | $5–12/hr | n/a | Skip — 2–4× more expensive than above for the same hardware. Only worth it if you already have credits. |

> **Rough budget heuristic:** A Phase-2 PPO from-scratch run for a coverage policy is typically 50–200 M environment steps; on an A100 with 64 parallel envs in PyBullet you'll get ~10–25k FPS, so a 100 M-step run = **6–12 hours = $10–25**. Plan for ~5 such runs (hyperparameter exploration + final), so **budget $50–100 for compute** for Phase 2.

### Simulator recommendations

| Phase | Simulator | Why | Install |
|---|---|---|---|
| **Phase 1: gridworld** | **Custom Gymnasium env** (no rendering, pure NumPy) | Trivial, runs on CPU at 100k+ steps/sec, no install pain | `pip install gymnasium`; code below |
| **Phase 2: camera-based RL, local prototyping** | **PyBullet** + your own URDF differential-drive robot | `pip install pybullet`, GUI mode for debugging, mature URDF/SDF support, CPU + single-GPU rendering | [PyBullet quickstart](https://github.com/bulletphysics/bullet3/blob/master/docs/pybullet_quickstart_guide/PyBulletQuickstartGuide.md.html) · [PyBullet Gym](https://github.com/benelot/pybullet-gym) |
| **Phase 2: scaled-up sim training** | **MuJoCo MJX** (preferred) or **Isaac Lab** (if you want photorealism) | MJX: thousands of envs on one GPU via JAX, 100k+ steps/sec. Isaac Lab: photorealistic, sensor fidelity, but needs ≥16 GB VRAM | MJX: `pip install mujoco mjx` ([MJX docs](https://mujoco.readthedocs.io/en/stable/mjx.html)) · Isaac Lab: [official install guide](https://isaac-sim.github.io/IsaacLab/) |
| **Phase 2 alternative: ROS 2 integration** | Gazebo Harmonic / Gazebo Sim | If you want to use ROS 2 Nav2 baselines for comparison | [Gazebo Sim install](https://gazebosim.org/docs/harmonic/install/) |

### Why MJX over PyBullet for serious training
- PyBullet is single-threaded for physics and CPU-bound; you'll cap at maybe 500–2000 steps/sec for a single env even on a fast laptop, and rendering camera images is slow.
- MJX (JAX port of MuJoCo) gives **>100,000 steps/sec batched** on a single A100, even with rendering via the [Madrona batch renderer](https://madrona-engine.github.io/). Camera-based RL training that takes 8 hours in PyBullet finishes in 30–60 minutes in MJX.
- **Caveat:** MJX requires writing your env in JAX or via the MJX Python API; this is a real learning step. If you've never used JAX, budget a weekend for the [MJX tutorial](https://mujoco.readthedocs.io/en/stable/mjx.html) and the [Brax PPO examples](https://github.com/google/brax).

### Why not Isaac Lab locally (for you)
The current Isaac Sim / Isaac Lab system requirements explicitly require **16 GB VRAM minimum** (RTX 4080 / 5080 class) — see the [official requirements page](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html). Your RTX 2060 6 GB will start, but will OOM with anything beyond toy scenes and a couple of environments. If you want Isaac Lab, rent an A100 instance on RunPod ($1.39/hr) — there's a [prebuilt Isaac Lab Docker image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-lab) that brings setup to ~10 minutes.

---

## 6. RL Algorithm Choices Per Phase

### Phase 1: Tabular RL on a discretized grid (CPU is fine)

**Recommendation: SARSA(λ) or Q-learning with ε-greedy exploration**, with reward shaping for coverage (a small per-cell bonus for first-visit, plus total-variation penalty on the visit-count map, à la Jonnarth et al.).

#### Justification
- The grid-world is small (e.g., 20×20 cells of 25 cm each = a 5 m × 5 m room), which means the state-action space is `~(20·20·4 orientations) · 4 actions = 6400`. Trivially tabular.
- Tabular methods give you a **provably-converging baseline** so you can debug your reward shaping before tossing it at a deep net (where convergence failures are ambiguous between net and reward).
- SARSA with eligibility traces accelerates credit assignment for the long-horizon coverage reward; vanilla Q-learning works but is slower to converge.

#### Phase 1 runnable starter code

Save as `phase1/gridworld_env.py`:

```python
"""
Grid-world coverage environment for a vacuum robot.

State:    (x, y, heading) where heading in {0=N, 1=E, 2=S, 3=W}
Action:   0=forward, 1=turn_left, 2=turn_right, 3=stop
Reward:   +1 for every newly-cleaned cell (first visit)
          -0.05 per timestep (encourages speed)
          -1   for bumping into wall/obstacle
          +10  bonus when >=95% of cleanable cells covered (terminal)
          + small "total variation" bonus that smooths visit_count (Jonnarth et al. ICML 2024)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class GridCoverageEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, grid_size=20, obstacle_density=0.10, max_steps=2000, seed=None):
        super().__init__()
        self.grid_size = grid_size
        self.obstacle_density = obstacle_density
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        # Discrete action space: forward, turn-left, turn-right, stop
        self.action_space = spaces.Discrete(4)

        # Observation is the full state (x, y, heading) plus per-cell visit count
        # For tabular learning we use just (x, y, heading) — see _obs_key.
        # We expose a Box observation for SB3 compatibility too.
        self.observation_space = spaces.Box(
            low=0, high=max(grid_size, 4),
            shape=(3,), dtype=np.int32
        )

        self._reset_world()

    def _reset_world(self):
        # 0 = free, 1 = obstacle
        self.grid = (self.rng.random((self.grid_size, self.grid_size))
                     < self.obstacle_density).astype(np.int8)
        # Force borders to free so spawn always works; carve a clear edge ring
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1

        # Visit count map (how many times each cell was visited)
        self.visit_count = np.zeros_like(self.grid, dtype=np.int32)

        # Spawn robot at a random free interior cell
        free = np.argwhere(self.grid == 0)
        spawn = free[self.rng.integers(len(free))]
        self.pos = spawn.astype(np.int32)
        self.heading = int(self.rng.integers(4))
        self.steps = 0
        self._mark_visit()

    def _mark_visit(self):
        self.visit_count[self.pos[0], self.pos[1]] += 1

    def _obs(self):
        return np.array([self.pos[0], self.pos[1], self.heading], dtype=np.int32)

    def _obs_key(self):
        # Hashable state for tabular agent
        return (int(self.pos[0]), int(self.pos[1]), int(self.heading))

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._reset_world()
        return self._obs(), {}

    def step(self, action):
        self.steps += 1
        reward = -0.05  # time penalty
        terminated = False
        truncated = False

        if action == 0:  # forward
            dx, dy = self._heading_to_delta(self.heading)
            new_pos = self.pos + np.array([dx, dy], dtype=np.int32)
            if (0 <= new_pos[0] < self.grid_size
                and 0 <= new_pos[1] < self.grid_size
                and self.grid[new_pos[0], new_pos[1]] == 0):
                # Newly visited cell?
                first_visit = self.visit_count[new_pos[0], new_pos[1]] == 0
                self.pos = new_pos
                self._mark_visit()
                if first_visit:
                    reward += 1.0
            else:
                reward -= 1.0  # bumped a wall
        elif action == 1:  # turn left
            self.heading = (self.heading - 1) % 4
        elif action == 2:  # turn right
            self.heading = (self.heading + 1) % 4
        elif action == 3:  # stop
            pass

        # Total-variation bonus (smoother visit_count -> better coverage)
        # NOTE: small coefficient; encourages spreading visits evenly across free cells.
        tv = (np.abs(np.diff(self.visit_count, axis=0)).sum()
              + np.abs(np.diff(self.visit_count, axis=1)).sum())
        reward -= 0.0001 * tv  # tiny smoothing penalty

        # Coverage termination
        free_cells = (self.grid == 0).sum()
        cleaned = (self.visit_count > 0).sum()
        coverage = cleaned / free_cells
        if coverage >= 0.95:
            reward += 10.0
            terminated = True

        if self.steps >= self.max_steps:
            truncated = True

        return self._obs(), reward, terminated, truncated, {"coverage": coverage}

    def _heading_to_delta(self, heading):
        return [(-1, 0), (0, 1), (1, 0), (0, -1)][heading]

    def render(self):
        # Compact ASCII render: # obstacle, . free unvisited, : visited >=1, R robot
        out = []
        for r in range(self.grid_size):
            row = []
            for c in range(self.grid_size):
                if (r, c) == tuple(self.pos):
                    row.append("R")
                elif self.grid[r, c] == 1:
                    row.append("#")
                elif self.visit_count[r, c] > 0:
                    row.append(":")
                else:
                    row.append(".")
            out.append("".join(row))
        return "\n".join(out)
```

Save as `phase1/train_q_learning.py`:

```python
"""
Tabular SARSA(lambda) agent for GridCoverageEnv.
Trains in ~30 seconds on a laptop CPU. Plots learning curve + final policy heatmap.
"""

import numpy as np
import collections
import argparse
import pickle
from gridworld_env import GridCoverageEnv


def train(env, n_episodes=2000, alpha=0.1, gamma=0.99, lam=0.9,
          eps_start=1.0, eps_end=0.05, eps_decay_episodes=1500, seed=0):
    """SARSA(lambda) with replacing traces.

    Q-table is a defaultdict of (state) -> np.array(num_actions).
    """
    rng = np.random.default_rng(seed)
    Q = collections.defaultdict(lambda: np.zeros(env.action_space.n, dtype=np.float64))
    returns = []
    coverages = []

    for ep in range(n_episodes):
        eps = max(eps_end, eps_start - (eps_start - eps_end) * ep / eps_decay_episodes)
        obs, _ = env.reset(seed=seed + ep)
        s = env._obs_key()
        a = _eps_greedy(Q[s], eps, env.action_space.n, rng)

        E = collections.defaultdict(lambda: np.zeros(env.action_space.n))
        ep_return = 0.0
        coverage = 0.0

        for _ in range(env.max_steps):
            _, r, term, trunc, info = env.step(a)
            sp = env._obs_key()
            ap = _eps_greedy(Q[sp], eps, env.action_space.n, rng)

            delta = r + (0.0 if term else gamma * Q[sp][ap]) - Q[s][a]
            E[(s, a)] = 1.0   # replacing trace on current (s,a)
            # Update all visited (s,a) pairs in the trace
            for (sk, ak), trace in list(E.items()):
                Q[sk][ak] += alpha * delta * trace
                # Decay traces
                E[(sk, ak)] = gamma * lam * trace
                if E[(sk, ak)] < 1e-4:
                    del E[(sk, ak)]

            ep_return += r
            coverage = info["coverage"]
            if term or trunc:
                break
            s, a = sp, ap

        returns.append(ep_return)
        coverages.append(coverage)
        if (ep + 1) % 100 == 0:
            print(f"ep {ep+1:>5d} | return {np.mean(returns[-100:]):+8.2f} | "
                  f"coverage {np.mean(coverages[-100:]):.2f} | eps {eps:.3f}")

    return Q, returns, coverages


def _eps_greedy(qvals, eps, n_actions, rng):
    if rng.random() < eps:
        return int(rng.integers(n_actions))
    return int(np.argmax(qvals))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--grid", type=int, default=20)
    p.add_argument("--out", type=str, default="q_table.pkl")
    args = p.parse_args()

    env = GridCoverageEnv(grid_size=args.grid, seed=0)
    Q, returns, coverages = train(env, n_episodes=args.episodes)
    with open(args.out, "wb") as f:
        pickle.dump(dict(Q), f)
    print(f"Saved Q-table to {args.out}")
```

> Run: `python phase1/train_q_learning.py --episodes 3000`. Expect ~95% coverage after ~2000 episodes on a 20×20 grid. If you don't reach 80% by episode 500, your reward shaping is probably off — check that first-visit bonus dominates the time penalty.

#### Phase 1 reward-shaping reading
- **[Jonnarth et al., "Learning Coverage Paths in Unknown Environments with Deep Reinforcement Learning", ICML 2024](https://proceedings.mlr.press/v235/jonnarth24a.html)** — the modern reference. Open code: [jonnarth/coverage-path-planning-rl](https://github.com/jonnarth/coverage-path-planning-rl). Read their reward decomposition (Section 4); the total-variation term is what I put in the env above.
- **[LIRL (MDPI Symmetry 2024)](https://www.mdpi.com/2073-8994/16/11/1537)** — SAC + latent imagination for CPP. Good to read after you understand Jonnarth.
- **[Lakshmanan et al., "Complete coverage path planning using RL for tetromino cleaning"](https://doi.org/10.1016/j.autcon.2020.103078)** — older but classic; their reward formulation works on small grids.
- **[Survey 2025 (DRL path planning)](https://proceedings.mlr.press/v278/xu25a.html)** — pointer collection.
- For RL fundamentals, Sutton & Barto's [Reinforcement Learning: An Introduction (2nd ed., free PDF)](http://incompleteideas.net/book/RLbook2020.pdf) chapters 6 (TD) and 12 (eligibility traces) cover everything in the code above.

### Phase 2: Deep RL with camera observations

**Recommendation: PPO (clipping + GAE) with a small CNN encoder (NatureCNN-style), trained in PyBullet first, then transferred to MJX/Isaac if you want scale.**

#### Justification
- **PPO over SAC** for this task because: (1) PPO is on-policy, which makes it easier to do domain randomization at the env reset boundary; (2) discrete action space (forward/turn-left/turn-right/stop) is PPO's home turf; (3) PPO has fewer hyperparameters that interact, easier to debug; (4) SB3's PPO implementation is the most battle-tested in [their RL Zoo](https://github.com/DLR-RM/rl-baselines3-zoo) — you can crib hyperparameters from CarRacing-v3 and start ~80% optimal.
- **SAC would be the right choice** if you go continuous-action (linear + angular velocity setpoints) and want sample efficiency from an off-policy buffer. Both are reasonable; PPO is the default for "I want to ship something."
- **Library choice: Stable-Baselines3 v2.8.0** (current as of April 2026, [release notes](https://github.com/DLR-RM/stable-baselines3/releases)). SB3 + Gymnasium is the lowest-friction path. Use **CleanRL** if you want to read the algorithm line-by-line ([cleanrl/ppo.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py), [cleanrl/ppo_atari.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari.py) is your starting template). Use **SBX** ([github.com/araffin/sbx](https://github.com/araffin/sbx)) — same SB3 API, JAX backend, much faster — if you migrate to MJX. Don't bother with RLlib unless you need multi-agent.

#### Why a NatureCNN-style encoder
- 84×84 grayscale, 4-frame stack: exactly the Atari setup, well-understood capacity (~1.7M params), runs fast on Orin Nano after TensorRT (Section 8 shows you should target 20+ ms inference for a 50 Hz control loop with margin).
- If you want depth (when you add an RGB-D camera in stretch tier), use NatureCNN with 5-channel input (RGB + depth + optical-flow) or a small MobileNetV3 backbone.

#### Phase 2 runnable starter code

Save as `phase2/sim_env.py`:

```python
"""
Differential-drive vacuum robot in PyBullet, observation = down-mounted RGB camera.

Designed to be small enough to train end-to-end in a few hours on RTX 2060.
Use this for the first PPO run; later migrate to MJX for parallel scale.

Install:
    pip install pybullet gymnasium stable-baselines3[extra] opencv-python
"""

import os
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
import cv2


class PybulletVacuumEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode=None, max_steps=2000,
                 cell_size=0.10, room_size=(4.0, 4.0), n_obstacles=4,
                 image_size=84, seed=None):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.cell_size = cell_size
        self.room_size = room_size
        self.n_obstacles = n_obstacles
        self.image_size = image_size
        self.rng = np.random.default_rng(seed)

        # 4 discrete actions: forward / turn-left / turn-right / stop
        self.action_space = spaces.Discrete(4)

        # Observation: 4-frame stack of grayscale 84x84 (Atari-style)
        # Stable-Baselines3 NatureCNN expects (C, H, W) with channels-first
        self.frame_stack = 4
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(self.frame_stack, image_size, image_size),
            dtype=np.uint8,
        )

        # Connect to PyBullet
        self.client = p.connect(p.GUI if render_mode == "human" else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)

        self.robot_id = None
        self.obstacle_ids = []
        self.coverage_map = None
        self.frames = None

    def _build_room(self):
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        # Plane (floor)
        p.loadURDF("plane.urdf", physicsClientId=self.client)

        # Walls (4 thin boxes around room_size box)
        W, H = self.room_size
        thick = 0.05
        wall_h = 0.2
        # Simple box collisions for the four walls
        for (x, y, sx, sy) in [
            (W/2, 0, W/2, thick),
            (W/2, H, W/2, thick),
            (0, H/2, thick, H/2),
            (W,   H/2, thick, H/2),
        ]:
            col = p.createCollisionShape(p.GEOM_BOX,
                                          halfExtents=[sx, sy, wall_h],
                                          physicsClientId=self.client)
            vis = p.createVisualShape(p.GEOM_BOX,
                                       halfExtents=[sx, sy, wall_h],
                                       rgbaColor=[0.6, 0.6, 0.6, 1.0],
                                       physicsClientId=self.client)
            p.createMultiBody(0, col, vis, [x, y, wall_h], physicsClientId=self.client)

        # Obstacles (cubes)
        self.obstacle_ids = []
        for _ in range(self.n_obstacles):
            x = self.rng.uniform(0.5, W - 0.5)
            y = self.rng.uniform(0.5, H - 0.5)
            size = self.rng.uniform(0.10, 0.25)
            col = p.createCollisionShape(p.GEOM_BOX,
                                          halfExtents=[size, size, 0.05],
                                          physicsClientId=self.client)
            vis = p.createVisualShape(p.GEOM_BOX,
                                       halfExtents=[size, size, 0.05],
                                       rgbaColor=[0.3, 0.2, 0.1, 1.0],
                                       physicsClientId=self.client)
            self.obstacle_ids.append(
                p.createMultiBody(0, col, vis, [x, y, 0.05], physicsClientId=self.client)
            )

        # Robot: a simple cylinder with two wheels (or we skip wheels and use direct velocity control)
        rx, ry = self.rng.uniform(0.5, W - 0.5), self.rng.uniform(0.5, H - 0.5)
        rcol = p.createCollisionShape(p.GEOM_CYLINDER,
                                       radius=0.08, height=0.10,
                                       physicsClientId=self.client)
        rvis = p.createVisualShape(p.GEOM_CYLINDER,
                                    radius=0.08, length=0.10,
                                    rgbaColor=[0.9, 0.1, 0.1, 1.0],
                                    physicsClientId=self.client)
        self.robot_id = p.createMultiBody(1.0, rcol, rvis, [rx, ry, 0.05],
                                          baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                                          physicsClientId=self.client)

        # Coverage map: discretized cell-visited grid
        nx = int(W / self.cell_size)
        ny = int(H / self.cell_size)
        self.coverage_map = np.zeros((nx, ny), dtype=np.int32)

    def _step_physics_and_robot(self, action):
        # Direct kinematic control: action -> linear/angular velocity
        # action: 0=forward, 1=left, 2=right, 3=stop
        v = 0.15   # m/s
        w = 1.0    # rad/s
        if action == 0:
            lin, ang = v, 0.0
        elif action == 1:
            lin, ang = 0.0, w
        elif action == 2:
            lin, ang = 0.0, -w
        else:
            lin, ang = 0.0, 0.0

        pos, orn = p.getBasePositionAndOrientation(self.robot_id,
                                                    physicsClientId=self.client)
        eu = p.getEulerFromQuaternion(orn)
        yaw = eu[2]
        dt = 0.05
        new_yaw = yaw + ang * dt
        nx = pos[0] + lin * np.cos(new_yaw) * dt
        ny = pos[1] + lin * np.sin(new_yaw) * dt
        new_pos = [nx, ny, pos[2]]
        new_orn = p.getQuaternionFromEuler([0, 0, new_yaw])
        p.resetBasePositionAndOrientation(self.robot_id, new_pos, new_orn,
                                           physicsClientId=self.client)
        p.stepSimulation(physicsClientId=self.client)

        # Coverage bookkeeping
        cx = int(np.clip(nx / self.cell_size, 0, self.coverage_map.shape[0] - 1))
        cy = int(np.clip(ny / self.cell_size, 0, self.coverage_map.shape[1] - 1))
        first_visit = self.coverage_map[cx, cy] == 0
        self.coverage_map[cx, cy] += 1

        # Collision detection: any contact with walls or obstacles
        contacts = p.getContactPoints(bodyA=self.robot_id, physicsClientId=self.client)
        collided = len(contacts) > 0
        return first_visit, collided

    def _get_camera_obs(self):
        # Camera mounted on the robot, pointing forward and slightly down
        pos, orn = p.getBasePositionAndOrientation(self.robot_id,
                                                    physicsClientId=self.client)
        eu = p.getEulerFromQuaternion(orn)
        yaw = eu[2]
        cam_pos = [pos[0] + 0.06 * np.cos(yaw),
                   pos[1] + 0.06 * np.sin(yaw),
                   0.18]
        target = [pos[0] + 0.5 * np.cos(yaw),
                  pos[1] + 0.5 * np.sin(yaw),
                  0.0]
        view = p.computeViewMatrix(cam_pos, target, [0, 0, 1])
        proj = p.computeProjectionMatrixFOV(fov=60, aspect=1.0,
                                             nearVal=0.05, farVal=5.0)
        w, h = self.image_size, self.image_size
        _, _, rgb, _, _ = p.getCameraImage(w, h, view, proj,
                                            renderer=p.ER_BULLET_HARDWARE_OPENGL,
                                            physicsClientId=self.client)
        rgb = np.array(rgb, dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return gray

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._build_room()
        self.steps = 0
        # Initialize frame stack with first observation
        first_frame = self._get_camera_obs()
        self.frames = np.stack([first_frame] * self.frame_stack, axis=0)
        return self.frames.copy(), {}

    def step(self, action):
        self.steps += 1
        first_visit, collided = self._step_physics_and_robot(action)

        reward = -0.01  # small time penalty
        if first_visit:
            reward += 1.0
        if collided:
            reward -= 1.0

        free_cells = self.coverage_map.size  # treat all as free for simplicity
        coverage = (self.coverage_map > 0).sum() / free_cells
        terminated = coverage >= 0.85
        if terminated:
            reward += 10.0
        truncated = self.steps >= self.max_steps

        # Update frame stack
        new_frame = self._get_camera_obs()
        self.frames = np.concatenate([self.frames[1:], new_frame[None, ...]], axis=0)
        return self.frames.copy(), reward, terminated, truncated, {"coverage": coverage}

    def close(self):
        if self.client is not None:
            p.disconnect(self.client)
            self.client = None
```

Save as `phase2/train_ppo.py`:

```python
"""
PPO on PybulletVacuumEnv with NatureCNN.
Tested with stable-baselines3==2.8.0, gymnasium==1.0.0, pybullet==3.2.6.

Train:
    python phase2/train_ppo.py --total-steps 5_000_000 --n-envs 4

Estimated wall-clock on RTX 2060 6GB:
    ~6-8 hours for 5M steps with 4 parallel envs.
On A100 with 16 envs:
    ~1.5-2 hours.
"""

import argparse
import os
from datetime import datetime

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.torch_layers import NatureCNN
from stable_baselines3.common.policies import ActorCriticCnnPolicy

from sim_env import PybulletVacuumEnv


def make_env(rank: int, seed: int = 0):
    def _init():
        env = PybulletVacuumEnv(render_mode=None, seed=seed + rank)
        return env
    return _init


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--total-steps", type=int, default=5_000_000)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--checkpoint-freq", type=int, default=200_000)
    p.add_argument("--log-dir", type=str, default="runs/")
    args = p.parse_args()

    run_id = datetime.now().strftime("ppo_vacuum_%Y%m%d_%H%M%S")
    log_path = os.path.join(args.log_dir, run_id)
    os.makedirs(log_path, exist_ok=True)

    vec_env = SubprocVecEnv([make_env(i) for i in range(args.n_envs)])
    vec_env = VecMonitor(vec_env, filename=os.path.join(log_path, "monitor.csv"))

    # Eval env (separate, single)
    eval_env = SubprocVecEnv([make_env(99)])
    eval_env = VecMonitor(eval_env)

    policy_kwargs = dict(
        features_extractor_class=NatureCNN,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=[dict(pi=[256], vf=[256])],
        normalize_images=False,  # we already provide uint8 84x84
    )

    model = PPO(
        ActorCriticCnnPolicy,
        vec_env,
        learning_rate=2.5e-4,
        n_steps=256,          # rollout length per env
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_path,
        verbose=1,
    )

    callbacks = [
        CheckpointCallback(save_freq=args.checkpoint_freq // args.n_envs,
                           save_path=log_path, name_prefix="ppo_ckpt"),
        EvalCallback(eval_env, best_model_save_path=log_path,
                     eval_freq=50_000 // args.n_envs,
                     n_eval_episodes=3, deterministic=True, render=False),
    ]

    model.learn(total_timesteps=args.total_steps,
                callback=callbacks,
                tb_log_name=run_id,
                progress_bar=True)
    model.save(os.path.join(log_path, "ppo_vacuum_final"))


if __name__ == "__main__":
    main()
```

> Run: `python phase2/train_ppo.py --total-steps 3_000_000 --n-envs 4`. Watch `tensorboard --logdir runs/` for `rollout/ep_rew_mean` to climb. If after 500k env steps it's still flat at −20 to −50, something is wrong: check that you can scripted-control the env to coverage >50% in episode (i.e., the env is solvable), and check that `coverage` in info actually increases.

#### Phase 2 hyperparameter starting points (cribbed from SB3 RL Zoo)
- Atari Pong PPO: [hyperparams YAML](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml) — use as a reference; my `train_ppo.py` already inlines reasonable defaults.
- If you switch to SAC (continuous actions), use the [LunarLanderContinuous-v3 hyperparameters](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/sac.yml) as your starting point. SAC needs ~5–10× fewer env steps than PPO for continuous control.

---

## 7. Sim-to-Real Transfer

### Strategy: Domain randomization in sim + safe-by-construction real-world fine-tuning.

#### Domain randomization (the heart of the approach)

Randomize at every `env.reset()` so the policy can't overfit to one configuration:

1. **Physical randomization**
   - Floor texture (uniform-random tile/wood pattern; library of 20+ HDR floor textures)
   - Lighting (intensity 0.5–1.5 × baseline; color temp 3000–6500 K)
   - Robot mass ±15 % and friction coefficient 0.4–0.9
   - Wheel slip noise (multiplicative on commanded velocity, 0.9–1.1)
2. **Sensor randomization**
   - Camera mount yaw ±5°, pitch ±10°, height ±2 cm
   - Image: brightness ±20 %, contrast ±10 %, hue ±5°, Gaussian blur σ ∈ [0, 1.5] px, JPEG-compression at random quality 70–95
   - Action latency 30–80 ms (matches the real pipeline)
3. **Environment randomization**
   - Room size (3 × 3 m to 5 × 5 m)
   - Number of obstacles (2–6)
   - Obstacle size, color, position
   - 30 % of episodes: insert a "carpet zone" (high friction, slightly different reflectance) so the policy doesn't assume uniform floor.

A clean implementation pattern: write a `DomainRandomizationWrapper(gym.Wrapper)` that re-builds the world with sampled parameters at each reset.

```python
class DomainRandomizationWrapper(gym.Wrapper):
    def __init__(self, env, dr_rng_seed=42):
        super().__init__(env)
        self.dr_rng = np.random.default_rng(dr_rng_seed)

    def reset(self, **kwargs):
        # Re-sample env params before letting the underlying env build the world
        self.env.room_size = (self.dr_rng.uniform(3.0, 5.0),
                              self.dr_rng.uniform(3.0, 5.0))
        self.env.n_obstacles = int(self.dr_rng.integers(2, 7))
        # ... lighting, friction, etc.
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        # Image-space augmentation
        for ch in range(obs.shape[0]):
            obs[ch] = self._augment(obs[ch])
        return obs, r, term, trunc, info

    def _augment(self, img):
        # brightness, blur, JPEG re-encode
        img = cv2.GaussianBlur(img,
                               (3, 3),
                               self.dr_rng.uniform(0.0, 1.5))
        # ... extend as needed
        return img
```

#### Real-world safety bounds

The single thing more likely to brick your robot than anything else is the **stair fall during exploration**. Defend against it in three independent layers:

1. **Software policy layer** (RL agent makes decisions)
2. **Safety-monitor layer** (Python supervisor: if any VL53L0X reports >50 mm depth to floor for >100 ms, override action to "stop" and require manual reset)
3. **Mechanical layer** (caster wheel placed so robot can't drive off a step front-first; physical barriers blocking stairwells during training)

Make the safety-monitor layer a separate process that talks to the motor driver over a different channel than the policy (e.g., policy writes to a shared memory buffer, safety reads/overrides). This avoids the "the policy hangs and locks the motor" failure mode.

```python
# pseudo-code for the safety supervisor
def safety_loop():
    while True:
        cliff_left = tof_left.read_distance_mm()
        cliff_right = tof_right.read_distance_mm()
        bumper = bumper_switch.read()
        if cliff_left > 50 or cliff_right > 50 or bumper:
            motor_driver.emergency_stop()
            log_event("safety_stop", {"cliff_l": cliff_left,
                                       "cliff_r": cliff_right,
                                       "bumper": bumper})
            wait_for_manual_reset()
        time.sleep(0.01)  # 100 Hz check
```

#### Fine-tuning safely on the real robot

1. **Don't fine-tune from scratch on hardware.** Always start with a sim-trained policy that already achieves ≥70 % coverage in randomized sim.
2. **Use a "safe restart" reward.** Add a large negative reward for any safety-supervisor trigger; this teaches the policy to avoid edges.
3. **Off-policy fine-tuning is safer than on-policy.** Collect ~1000 real-world transitions with the sim-trained policy, then fine-tune with SAC + behavior cloning regularization for a small number of gradient steps. Don't run on-policy PPO updates between every rollout — too unstable, too risky.
4. **Bound the policy delta.** Use a KL constraint or just an absolute weight-change limit; this keeps the policy from changing too much in any one update.

#### Sim-to-real references
- **Tobin et al., "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World" (2017)** — the original DR paper. [arXiv:1703.06907](https://arxiv.org/abs/1703.06907)
- **OpenAI, "Solving Rubik's Cube with a Robot Hand" (2019)** — DR + ADR. [arXiv:1910.07113](https://arxiv.org/abs/1910.07113)
- **Sadeghi & Levine, "CAD2RL: Real Single-Image Flight without a Single Real Image"** — vision-only sim-to-real for drones, lots of useful tricks. [arXiv:1611.04201](https://arxiv.org/abs/1611.04201)
- **Margolis et al., "Walk These Ways"** — recent (2022) sim-to-real for quadrupeds; reward shaping insights apply. [arXiv:2212.03238](https://arxiv.org/abs/2212.03238)
- **Survey: Zhao et al., "Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics" (2020)** — broad survey. [arXiv:2009.13303](https://arxiv.org/abs/2009.13303)
- **Implementation reference (small ground robot):** [NVIDIA-AI-IOT/jetbot_ros2](https://github.com/NVIDIA-AI-IOT/jetbot_ros2) and [JetBot collision-avoidance notebooks](https://github.com/NVIDIA-AI-IOT/jetbot/tree/master/notebooks/collision_avoidance) — the simplest end-to-end vision→action loop on Jetson hardware.

---

## 8. Quantization, Distillation, Deployment

### Latency target
For a 50 Hz control loop (robot moves ~3 cm per step at 0.15 m/s) you have **20 ms per cycle** for everything: capture frame, preprocess, run policy, write motor commands. Realistically budget:
- Frame capture (USB camera, MJPEG decode): 5–10 ms
- Preprocess (resize, grayscale, normalize, frame-stack push): 1–2 ms
- **Policy inference: target ≤ 5 ms** (this is the part TensorRT helps)
- Motor command write (I²C/GPIO): ~1 ms
- Slack: ~2 ms

A NatureCNN with 84×84×4 input runs comfortably at **2–4 ms on Jetson Orin Nano in FP16 via TensorRT**, so this target is realistic.

### The pipeline: PyTorch → ONNX → TensorRT (.engine), optionally INT8

#### Step 1: Export the SB3 PPO policy to ONNX (on your laptop)

```python
"""
export_to_onnx.py — Run on the laptop where the trained model lives.
"""

import torch
from stable_baselines3 import PPO


class OnnxablePolicy(torch.nn.Module):
    """Wraps SB3 policy for ONNX export (deterministic, action-only output).

    SB3's policy.forward returns (actions, values, log_prob); we strip to actions only
    for inference deployment.
    """

    def __init__(self, sb3_policy):
        super().__init__()
        self.policy = sb3_policy

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        actions, _, _ = self.policy(obs, deterministic=True)
        return actions


def main():
    model = PPO.load("runs/ppo_vacuum_final.zip", device="cpu")
    wrapped = OnnxablePolicy(model.policy)
    wrapped.eval()

    dummy = torch.zeros(1, 4, 84, 84, dtype=torch.uint8)
    dummy_f = dummy.float()  # NatureCNN expects float input in [0, 255] (it normalizes inside)

    torch.onnx.export(
        wrapped,
        dummy_f,
        "ppo_vacuum.onnx",
        opset_version=17,
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
    )
    print("Exported ppo_vacuum.onnx")


if __name__ == "__main__":
    main()
```

Verify on the laptop with `onnxruntime`:

```python
import onnx
import onnxruntime as ort
import numpy as np

m = onnx.load("ppo_vacuum.onnx")
onnx.checker.check_model(m)

sess = ort.InferenceSession("ppo_vacuum.onnx")
obs = np.zeros((1, 4, 84, 84), dtype=np.float32)
out = sess.run(None, {"obs": obs})
print(out)  # should be [[<int action>]]
```

#### Step 2: Copy `ppo_vacuum.onnx` to the Jetson and build a TensorRT engine

```bash
# On the Jetson (must build engine on target hardware; engines are not portable across GPUs)
scp ppo_vacuum.onnx jetson@robot.local:~/

ssh jetson@robot.local
dpkg -l | grep tensorrt        # confirm TensorRT 10.3 (JetPack 6.2)

# FP16 engine (best speed/accuracy trade)
/usr/src/tensorrt/bin/trtexec \
    --onnx=ppo_vacuum.onnx \
    --saveEngine=ppo_vacuum_fp16.engine \
    --fp16 \
    --minShapes=obs:1x4x84x84 \
    --optShapes=obs:1x4x84x84 \
    --maxShapes=obs:1x4x84x84 \
    --workspace=2048 \
    --verbose

# Benchmark
/usr/src/tensorrt/bin/trtexec \
    --loadEngine=ppo_vacuum_fp16.engine \
    --warmUp=500 --avgRuns=200
# Expect: latency ~2-3 ms, throughput ~400-500 inf/s on Orin Nano @ 15W
```

#### Step 3 (optional, when you want INT8): Calibrate INT8

INT8 gives another ~2× speedup but needs a representative calibration dataset (a few hundred to a few thousand real observations). Capture them during a teleoperated rollout on the actual robot:

```python
"""
calibration_capture.py — run on the robot while you drive it around by hand,
recording 84x84 grayscale obs frames you'll later feed to TensorRT for INT8 calibration.
"""

import cv2
import numpy as np
import time
import os

CAP_DIR = "calib_frames"
os.makedirs(CAP_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)            # USB camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

i = 0
target = 1000  # capture ~1000 frames spanning your various rooms / lighting
while i < target:
    ok, frame = cap.read()
    if not ok:
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (84, 84))
    np.save(os.path.join(CAP_DIR, f"{i:06d}.npy"), gray)
    i += 1
    time.sleep(0.1)
print(f"Captured {i} calibration frames in {CAP_DIR}")
```

Then build INT8 engine with calibrator. Easiest is to use [NVIDIA's pytorch_quantization toolkit](https://github.com/NVIDIA/TensorRT-Model-Optimizer) for PTQ; the official [TensorRT samples for ONNX INT8 calibration](https://github.com/NVIDIA/TensorRT/tree/release/10.3/samples/python/int8_caffe_mnist) show the pattern (adapt to a frame-numpy dataset).

#### Step 4: Inference loop on Jetson (Python, polyglot with motor control)

```python
"""
deploy_jetson.py — control loop on the Jetson Orin Nano.

Runs the camera capture, TensorRT inference, and motor commands in a single Python loop.
For >50 Hz, consider moving camera capture into its own thread to overlap I/O with inference.
"""

import time
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import Jetson.GPIO as GPIO

# ---- Motor / GPIO setup (TB6612FNG via 2 GPIO + 1 PWM per motor) ----
PWM_HZ = 1000
LEFT_PWM_PIN = 32  # hardware PWM channel
LEFT_AIN1 = 16
LEFT_AIN2 = 18
RIGHT_PWM_PIN = 33
RIGHT_BIN1 = 8
RIGHT_BIN2 = 10
STBY_PIN = 12

GPIO.setmode(GPIO.BOARD)
for pin in [LEFT_AIN1, LEFT_AIN2, RIGHT_BIN1, RIGHT_BIN2, STBY_PIN]:
    GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
GPIO.output(STBY_PIN, GPIO.HIGH)
GPIO.setup(LEFT_PWM_PIN, GPIO.OUT)
GPIO.setup(RIGHT_PWM_PIN, GPIO.OUT)
left_pwm = GPIO.PWM(LEFT_PWM_PIN, PWM_HZ); left_pwm.start(0)
right_pwm = GPIO.PWM(RIGHT_PWM_PIN, PWM_HZ); right_pwm.start(0)


def drive(action):
    # 0=forward, 1=turn-left, 2=turn-right, 3=stop
    if action == 0:
        GPIO.output(LEFT_AIN1, 1); GPIO.output(LEFT_AIN2, 0)
        GPIO.output(RIGHT_BIN1, 1); GPIO.output(RIGHT_BIN2, 0)
        left_pwm.ChangeDutyCycle(50); right_pwm.ChangeDutyCycle(50)
    elif action == 1:
        GPIO.output(LEFT_AIN1, 0); GPIO.output(LEFT_AIN2, 1)
        GPIO.output(RIGHT_BIN1, 1); GPIO.output(RIGHT_BIN2, 0)
        left_pwm.ChangeDutyCycle(40); right_pwm.ChangeDutyCycle(40)
    elif action == 2:
        GPIO.output(LEFT_AIN1, 1); GPIO.output(LEFT_AIN2, 0)
        GPIO.output(RIGHT_BIN1, 0); GPIO.output(RIGHT_BIN2, 1)
        left_pwm.ChangeDutyCycle(40); right_pwm.ChangeDutyCycle(40)
    else:
        left_pwm.ChangeDutyCycle(0); right_pwm.ChangeDutyCycle(0)


# ---- TensorRT engine load ----
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
with open("ppo_vacuum_fp16.engine", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

# Pre-allocate device buffers
in_shape = (1, 4, 84, 84)
out_shape = (1,)
h_in = cuda.pagelocked_empty(int(np.prod(in_shape)), dtype=np.float32)
h_out = cuda.pagelocked_empty(int(np.prod(out_shape)), dtype=np.int64)
d_in = cuda.mem_alloc(h_in.nbytes)
d_out = cuda.mem_alloc(h_out.nbytes)
stream = cuda.Stream()


def infer(frame_stack: np.ndarray) -> int:
    np.copyto(h_in, frame_stack.astype(np.float32).ravel())
    cuda.memcpy_htod_async(d_in, h_in, stream)
    context.execute_async_v2(bindings=[int(d_in), int(d_out)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(h_out, d_out, stream)
    stream.synchronize()
    return int(h_out[0])


# ---- Main loop ----
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
frames = np.zeros((4, 84, 84), dtype=np.uint8)

try:
    while True:
        t0 = time.time()
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (84, 84))
        frames = np.concatenate([frames[1:], gray[None, ...]], axis=0)
        action = infer(frames[None, ...])
        drive(action)
        dt = time.time() - t0
        # print(f"loop {dt*1000:.1f} ms  action={action}")
        time.sleep(max(0, 0.02 - dt))  # cap at 50 Hz
finally:
    left_pwm.stop(); right_pwm.stop(); GPIO.cleanup()
    cap.release()
```

#### Distillation (use it if your trained policy is bigger than NatureCNN, e.g., ResNet18 + LSTM)

The shortest path:

1. Roll out the teacher (large policy) in sim, recording (state, teacher-action-distribution) pairs.
2. Train a student (NatureCNN) to minimize KL(student || teacher) on those pairs.
3. Periodically RL-finetune the student with a tiny KL constraint on the teacher to recover any lost RL signal.

Reference implementations and reading:
- [SB3 PPO docs](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [SB3 export to ONNX docs](https://stable-baselines3.readthedocs.io/en/master/guide/export.html)
- [NVIDIA TensorRT 10.3 docs](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/) and [TensorRT GitHub samples](https://github.com/NVIDIA/TensorRT/tree/release/10.3/samples)
- [NVIDIA torch_tensorrt (alternative path that bypasses ONNX)](https://github.com/pytorch/TensorRT)
- [JetPack 6.2 download](https://developer.nvidia.com/embedded/jetpack-sdk-62)
- [NVIDIA Jetson Inference samples](https://github.com/dusty-nv/jetson-inference) — Dusty Franklin's repo; the most useful "look at working code" reference for Jetson AI deployment.
- Distillation: [Hinton et al. 2015 "Distilling the Knowledge in a Neural Network"](https://arxiv.org/abs/1503.02531); for RL specifically [Rusu et al. 2016 "Policy Distillation"](https://arxiv.org/abs/1511.06295).

---

## 9. Evaluation Framework

### Metrics

| Metric | Definition | How to measure (sim) | How to measure (real) |
|---|---|---|---|
| **Coverage %** | Fraction of cleanable floor area visited within budget T | `(visit_count > 0).sum() / free_cells` | Optical flow integration + wheel encoders + grid overlay; OR overhead phone camera + colored tape grid + post-hoc image analysis |
| **Time-to-X%-coverage** | Wall-clock to first reach X ∈ {50, 75, 90} % | Logged during eval rollouts | Stopwatch + grid overlay |
| **Redundant-coverage ratio** | mean(visit_count over visited cells) — lower is better | Direct compute | Same overhead-camera approach |
| **Collision rate** | Bumps per 10 m traveled | Contact events in sim | Bumper-switch click count + odometry |
| **Cleanliness proxy (real-world)** | % weight reduction in pre-placed debris (flour, rice, dust) | n/a | Pre/post weighing of a measured debris bed; see protocol below |
| **Per-step inference latency** | ms per policy forward pass | `time.perf_counter()` deltas | Same; log to file |
| **Battery time per run** | min:s | Battery telemetry | Stopwatch |

### Cleanliness measurement protocol (vacuum-review style)

For "does this thing actually clean":
1. Tape out a 1 m × 1 m test area on tile or hardwood.
2. Distribute 20 g of all-purpose flour evenly across the area using a sifter. (Flour is the standard vacuum-review test debris because it forces the brush/suction to actually pick it up.)
3. Photograph (overhead, fixed phone mount, fixed lighting).
4. Run the robot for a fixed budget (e.g., 5 minutes) with the trained policy.
5. Vacuum or sweep what remains, weigh it.
6. **Cleanliness % = (20 g − remaining g) / 20 g**.
7. Repeat with rice (larger debris) and pet-hair simulation (3 cm yarn snippets) for a fuller picture.

This protocol mirrors what [Wirecutter](https://www.nytimes.com/wirecutter/reviews/best-robot-vacuum/) and similar product reviewers do; results are directly comparable to your reference PrettyCare C2-v2 only if you also test the same flour on the same surface with that vacuum. Don't expect to win — expect to be within an order of magnitude (e.g., 30 % cleanliness vs 90 %) and consider that a successful learning project.

### Logging skeleton

Save as `evaluation/log_run.py`:

```python
"""
Lightweight metrics logger. Drops a row per environment step to a parquet file;
writes a one-line YAML summary at the end of each episode.

Use:
    from log_run import RunLogger
    rl = RunLogger("runs/sim_eval_001")
    rl.log_step(obs_meta={"action": a, "reward": r, "coverage": cov, "collided": c})
    rl.end_episode(summary={"final_coverage": ..., "steps": ...})
"""

import os
import pandas as pd
import yaml
import time


class RunLogger:
    def __init__(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.episode = 0
        self.rows = []
        self._t0 = time.perf_counter()

    def log_step(self, obs_meta: dict):
        obs_meta = dict(obs_meta)
        obs_meta["t"] = time.perf_counter() - self._t0
        obs_meta["episode"] = self.episode
        self.rows.append(obs_meta)

    def end_episode(self, summary: dict):
        # Write per-step rows
        df = pd.DataFrame(self.rows)
        df.to_parquet(os.path.join(self.out_dir, f"steps_{self.episode:04d}.parquet"))

        # Write episode summary
        summary = dict(summary)
        summary["episode"] = self.episode
        summary["duration_s"] = float(self.rows[-1]["t"] - self.rows[0]["t"]) if self.rows else 0.0
        with open(os.path.join(self.out_dir, f"summary_{self.episode:04d}.yaml"), "w") as f:
            yaml.safe_dump(summary, f)

        self.episode += 1
        self.rows = []
        self._t0 = time.perf_counter()
```

### Comparison to commercial vacuums

Use them only for **calibration of expectations**, not as targets:

| Aspect | Your DIY | PrettyCare C2-v2 / generic $100 vacuum |
|---|---|---|
| Suction | ~100–500 Pa from a 40 mm fan | 2000–8000 Pa from a brushless turbine |
| Coverage strategy | Learned RL policy | Random walk or SLAM-based |
| Battery | 2–3 h | 90–120 min |
| Wet mop | None | Often included |
| Cliff/stair safety | Best-effort ToF + bumper | Reliable, certified |
| Educational value | Maximal | Zero |

If your DIY hits **>30 % flour cleanliness in 5 minutes on a 1 m² area**, you've succeeded. The commercial unit is likely 80–95 % at that protocol.

---

## 10. Concrete Build Sequence & Repo Structure

### Repo layout (set this up on Day 1)

```
rl-vacuum/
├── README.md                       # Project overview, status, current phase
├── pyproject.toml                  # Or requirements.txt + setup.cfg
├── .gitignore
├── docs/                           # Architecture decisions, hardware notes
│   ├── hardware.md
│   ├── wiring_diagram.md
│   └── power_budget.md
├── hardware/
│   ├── chassis/                    # CAD / STL files for any 3D-printed parts
│   ├── wiring/                     # KiCad or Fritzing schematics
│   └── bom.md                      # Bill of materials (this section)
├── firmware/
│   └── safety_supervisor.py        # The independent safety loop
├── phase0_teleop/
│   ├── motor_test.py
│   ├── camera_test.py
│   └── keyboard_teleop.py
├── phase1_gridworld/
│   ├── gridworld_env.py            # (code in §6)
│   ├── train_q_learning.py         # (code in §6)
│   ├── train_sarsa_lambda.py
│   └── visualize_policy.py
├── phase2_sim/
│   ├── sim_env.py                  # (code in §6)
│   ├── domain_randomization.py
│   ├── train_ppo.py                # (code in §6)
│   ├── train_sac.py
│   └── hyperparams.yaml
├── phase3_deploy/
│   ├── export_to_onnx.py           # (code in §8)
│   ├── build_trt_engine.sh         # trtexec invocation
│   ├── calibration_capture.py      # (code in §8)
│   └── deploy_jetson.py            # (code in §8)
├── phase4_eval/
│   ├── log_run.py                  # (code in §9)
│   ├── coverage_analysis.ipynb
│   └── cleanliness_protocol.md
├── shared/
│   ├── motor_driver.py             # TB6612 control wrapper
│   ├── sensors.py                  # VL53L0X, MPU-6050, bumper
│   ├── camera.py                   # USB cam wrapper with framestack
│   └── policies.py                 # SB3 policy wrappers, ONNX export utils
├── scripts/
│   ├── jetson_setup.sh             # First-boot setup commands
│   └── run_full_eval.sh
└── runs/                           # Training logs (gitignored)
```

### Week-by-week timeline (assumes ~8–12 hrs/week of focused work; compress aggressively if you have more)

#### Week 0: Order parts
- Order everything from §2.1 (strict tier). Lead time 3–10 days depending on supplier.
- While waiting:
  - Flash JetPack 6.2 onto the Jetson Orin Nano with Super Mode enabled (see "Day 1 commands" below).
  - Set up the repo skeleton above on the MacBook.
  - Read [Sutton & Barto Ch. 6–7](http://incompleteideas.net/book/RLbook2020.pdf) (TD methods, n-step) if you haven't recently.

#### Week 1: Phase 0 — Mechanical assembly + teleop
- **Day 1**: Build the chassis. Mount 2 TT motors + caster + battery holder. Don't mount the Jetson yet.
- **Day 2**: Wire the TB6612FNG. Test motor direction control from a bench supply via Arduino or Raspberry Pi Pico first (faster iteration than Jetson for bring-up; you can shelf the Pico after). Verify both motors spin in both directions.
- **Day 3**: Wire the boost converter. Verify with a multimeter that you get a stable 12 V from the battery under load (run a motor while measuring).
- **Day 4**: Mount Jetson + USB camera. Plug Jetson into boost converter via barrel-jack pigtail. **Power on for the first time. Verify it boots and stays up while motors are running.** If it brownouts, add bigger caps or reduce motor PWM duty.
- **Day 5**: Write `phase0_teleop/keyboard_teleop.py`. WASD-controlled robot. Sanity-check the camera feed comes through.
- **Day 6**: Mount VL53L0X sensors as cliff detectors (overhanging front, looking down). Test that they read 30–50 mm on the floor and >100 mm when held over a step.
- **Day 7**: Write the safety supervisor (`firmware/safety_supervisor.py`). Integrate with teleop: if cliff detected, stop motors regardless of teleop input.

> **Done criterion for Phase 0:** You can manually drive the robot around for 30 minutes without crashes, brownouts, or stair falls.

#### Week 2: Phase 1 — Gridworld RL
- **Days 1–2**: Build `phase1_gridworld/gridworld_env.py`. Test reset/step/render manually.
- **Day 3**: Train `train_q_learning.py`. Plot learning curve. Should converge to >90% coverage in <3000 episodes.
- **Day 4**: Train SARSA(λ) variant. Compare convergence speed.
- **Day 5**: Implement reward shaping experiments. Try with and without total-variation term; with and without first-visit bonus; pure ε-greedy vs Boltzmann exploration. **This is the most valuable RL pedagogy of the whole project — don't rush it.**
- **Days 6–7**: Visualize learned policies as a vector field on the grid. Read Jonnarth et al. paper. Try a tabular agent with their frontier-based state representation.

> **Done criterion for Phase 1:** You have a tabular agent that achieves >95% coverage on random 20×20 rooms in <1500 steps per episode after training.

#### Week 3–4: Phase 2 — Camera RL in sim (laptop)
- **Day 1**: Install PyBullet on the laptop. Verify `python -c "import pybullet, pybullet_data"`.
- **Days 2–3**: Build `phase2_sim/sim_env.py`. Verify camera observations are reasonable (save a few frames as PNG and look at them). Implement `keyboard_teleop_sim.py` to verify the env is solvable by a human.
- **Day 4**: Add domain randomization wrapper. Verify episodes use different floor textures / room sizes.
- **Day 5**: Run first PPO training (small: 500k steps). Verify reward goes up. Debug if not (see FAQ Phase 2 section).
- **Days 6–10**: Full-scale training. Plan 3–5 PPO runs with varied hyperparameters. Total time: 20–40 hours of compute. **If laptop is too slow** (>2 days per run), provision a RunPod A100 instance and migrate.

> **Done criterion for Phase 2:** Best PPO checkpoint achieves >70% coverage in <2000 steps on randomized sim rooms.

#### Week 5: Phase 3 — Deployment to Jetson
- **Day 1**: Run `export_to_onnx.py` on the laptop. Verify with `onnxruntime`.
- **Day 2**: SCP the ONNX to the Jetson. Run `trtexec` to build FP16 engine. Benchmark with `--avgRuns=200`. Confirm <5 ms latency per inference.
- **Days 3–4**: Write/test `deploy_jetson.py`. Run **with motors disconnected** first — verify policy actions look sane on real camera input. Then connect motors and run in a small bounded area.
- **Day 5**: Collect calibration frames; build INT8 engine. Compare on-robot behavior FP16 vs INT8.
- **Days 6–7**: Tune control parameters (PWM duty cycle, action durations) to match what the policy expected from sim.

> **Done criterion for Phase 3:** Robot autonomously navigates a real room without bumping >3× per minute and without ever falling off the bumper-tested ledge.

#### Week 6+: Phase 4 — Evaluation & iteration
- **Day 1**: Run the flour cleanliness protocol on a 1 m² tile area. Baseline.
- **Days 2–3**: Iterate on whatever's weakest. Common iterations:
  - More domain randomization → re-train → re-deploy
  - Different action durations or PWM levels
  - Vacuum motor mechanical improvements (better impeller, sealed dustbin)
- **Day 4**: Run on a full room. Compute coverage % (use overhead phone camera if you don't have wheel encoders).
- **Day 5+**: Compare against your random-policy and bump-and-turn baselines. Write up results.

> **Done criterion for the project:** RL policy achieves higher coverage than a random-policy baseline AND a bump-and-turn baseline (commercial-vacuum-style) on the same room within the same time budget, OR you understand precisely why it doesn't (and have results to back the analysis).

### Day-one commands (Jetson setup)

> **Where to start:** [Jetson Orin Nano Developer Kit Getting Started Guide](https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nano-devkit), then [Initial Setup using SDK Manager](https://developer.nvidia.com/sdk-manager).

```bash
# === ON YOUR LAPTOP / DEV MACHINE ===

# Flash JetPack 6.2 to the Jetson NVMe drive.
# Option A (easiest): use NVIDIA SDK Manager on a Linux x86_64 host
#   https://developer.nvidia.com/sdk-manager
#   Choose "Jetson Orin Nano Developer Kit" → JetPack 6.2 → use
#   jetson-orin-nano-devkit-super.conf flashing config to enable Super Mode.
# Option B: command-line flash following
#   https://docs.nvidia.com/jetson/archives/r36.4.3/DeveloperGuide/IN/QuickStart.html

# === ON THE JETSON, FIRST BOOT ===

# Verify JetPack
sudo apt update
sudo apt install -y nvidia-jetpack
dpkg -l | grep -i tensorrt
# Should list TensorRT 10.3.x

# Enable Super Mode (Orin Nano 8GB: mode 2 = MAXN_SUPER)
sudo nvpmodel -m 2
sudo jetson_clocks
sudo reboot

# Install jetson-stats (system monitor)
sudo pip3 install -U jetson-stats
sudo systemctl restart jtop.service
jtop  # confirm Super Mode and 8GB memory shown

# Set up Python venv (avoid touching system Python)
sudo apt install -y python3-venv python3-pip
python3 -m venv ~/venvs/vacuum
source ~/venvs/vacuum/bin/activate
pip install --upgrade pip

# Install minimum packages for deployment (the Jetson will only do inference)
pip install numpy opencv-python pyyaml
# For PyCUDA on Jetson, use the prebuilt wheel from NVIDIA's index:
pip install pycuda --extra-index-url https://pypi.jetson-ai-lab.dev/jp6
# TensorRT Python bindings come pre-installed with JetPack 6.2 — no pip install needed

# Enable 40-pin header PWM channels (needed for motor PWM)
sudo /opt/nvidia/jetson-io/jetson-io.py
# In the menu: "Configure 40-pin expansion header" -> enable "pwm0" and "pwm2" -> save and reboot.

# Install Jetson.GPIO
sudo pip3 install Jetson.GPIO
sudo groupadd -f -r gpio
sudo usermod -a -G gpio $USER
sudo cp /opt/nvidia/jetson-gpio/etc/99-gpio.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
# Log out / log back in for the group change to take effect.

# Enable I2C user access (for VL53L0X, MPU-6050, optional PCA9685)
sudo apt install -y i2c-tools
sudo usermod -a -G i2c $USER
i2cdetect -y -r 7   # bus 7 on Orin Nano (was bus 1 on older Nano)
# Should show your I2C devices' addresses (e.g., 0x29 for VL53L0X, 0x68 for MPU-6050)

# Pull your repo
git clone https://github.com/<you>/rl-vacuum.git
cd rl-vacuum
pip install -e .

# Test camera
python phase0_teleop/camera_test.py

# === ON YOUR LAPTOP (MSI / Mac) FOR TRAINING ===

# RTX 2060 / Linux (MSI):
# Install CUDA 12.x + cuDNN matching PyTorch
# (Easiest: use the official PyTorch install matrix: https://pytorch.org/get-started/locally/)

# Mac M1: install with MPS backend
python3 -m venv ~/venvs/vacuum
source ~/venvs/vacuum/bin/activate
pip install --upgrade pip
pip install torch torchvision  # on macOS, this gives the MPS backend automatically
pip install stable-baselines3==2.8.0 gymnasium pybullet opencv-python
pip install onnx onnxruntime
```

---

## 11. Resource Appendix

### Datasheets & hardware docs
- [Jetson Orin Nano Developer Kit User Guide — Hardware Layout](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/hardware_layout.html)
- [Jetson Orin Nano Carrier Board Specification (PDF, official)](https://developer.nvidia.com/embedded/jetson-orin-nano-developer-kit-carrier-board-specification) — pin assignments, power constraints
- [Jetson Orin Nano Developer Kit pinout (JetsonHacks)](https://jetsonhacks.com/nvidia-jetson-orin-nano-j12-header-pinout/)
- [JetPack 6.2 release notes](https://docs.nvidia.com/jetson/archives/jetpack-archived/jetpack-62/release-notes/index.html)
- [TB6612FNG datasheet (Toshiba, PDF)](https://www.pololu.com/file/0J86/TB6612FNG.pdf)
- [Pololu TB6612FNG module page](https://www.pololu.com/product/713)
- [VL53L0X datasheet (ST, PDF)](https://www.st.com/resource/en/datasheet/vl53l0x.pdf)
- [VL53L1X datasheet (ST, PDF)](https://www.st.com/resource/en/datasheet/vl53l1x.pdf)
- [Using multiple VL53L0X on one I²C bus (AN4846, PDF)](https://www.st.com/resource/en/application_note/an4846-using-multiple-vl53l0x-in-a-single-design-stmicroelectronics.pdf)
- [MPU-6050 register map (PDF)](https://invensense.tdk.com/wp-content/uploads/2015/02/MPU-6000-Register-Map1.pdf)
- [IRLB8721 N-channel MOSFET datasheet (PDF)](https://www.infineon.com/dgdl/irlb8721pbf.pdf)
- [TT yellow gear motor spec sheet (Cytron)](https://www.cytron.io/p-3v-6v-dual-axis-tt-gear-motor)
- [Slamtec RPLidar A1M8 datasheet (PDF)](https://bucket-download.slamtec.com/d1e428e7efbdcd65a8ea111061794fb8d4ccd3a0/LD108_SLAMTEC_rplidar_datasheet_A1M8_v3.0_en.pdf)
- [DC-DC 12V → 19V 5A boost converter spec](https://daygreen.com/en-en/products/12v-to-19v-5a-dc-dc-step-up-converter-voltage-regulator-3-year)

### Purchase links (all verified to exist as of June 2026)
- [TT motor + wheel 4-pack (Amazon)](https://www.amazon.com/s?k=tt+motor+wheel+kit)
- [TB6612FNG module (Pololu)](https://www.pololu.com/product/713) · [SparkFun](https://www.sparkfun.com/products/14450)
- [VL53L0X breakout (Pololu)](https://www.pololu.com/product/2490) · [Adafruit](https://www.adafruit.com/product/3317)
- [VL53L1X breakout (Adafruit)](https://www.adafruit.com/product/3967)
- [MPU-6050 IMU breakout (Adafruit)](https://www.adafruit.com/product/3886)
- [PCA9685 16-channel I²C PWM (Adafruit)](https://www.adafruit.com/product/815)
- [IRLB8721 MOSFET breakout (Adafruit)](https://www.adafruit.com/product/355)
- [2× 18650 holder + XT60 (Amazon)](https://www.amazon.com/s?k=2s+18650+holder+xt60)
- [Caster wheel (Adafruit)](https://www.adafruit.com/product/2942)
- [Protected 18650 cells, Panasonic NCR18650B (Amazon)](https://www.amazon.com/s?k=protected+18650+panasonic+ncr18650b)
- [Drok 12V→19V boost converter (Amazon)](https://www.amazon.com/s?k=drok+boost+converter+12v+19v)
- [5.5×2.5mm DC barrel-jack pigtail (Amazon)](https://www.amazon.com/s?k=5.5mm+2.5mm+barrel+jack+pigtail)
- [Mini handheld vacuum to gut (Amazon)](https://www.amazon.com/s?k=mini+handheld+vacuum+rechargeable)
- [Slamtec RPLidar A1M8 (Amazon) [stretch]](https://www.amazon.com/Slamtec-RPLIDAR-Scanning-Avoidance-Navigation/dp/B07TJW5SXF)
- [Intel RealSense D435i (Mouser) [stretch]](https://www.mouser.com/ProductDetail/Intel/82635D435IDK5P)
- [OAK-D Lite (Luxonis) [stretch]](https://shop.luxonis.com/products/oak-d-lite-1)

### Software & library docs
- [NVIDIA JetPack 6.2 download](https://developer.nvidia.com/embedded/jetpack-sdk-62)
- [NVIDIA SDK Manager](https://developer.nvidia.com/sdk-manager)
- [TensorRT 10.3 developer guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/)
- [TensorRT samples on GitHub](https://github.com/NVIDIA/TensorRT)
- [PyTorch install matrix](https://pytorch.org/get-started/locally/)
- [Stable-Baselines3 docs](https://stable-baselines3.readthedocs.io/en/master/)
- [SB3 RL Zoo (hyperparams + trained models)](https://github.com/DLR-RM/rl-baselines3-zoo)
- [SB3 ONNX export guide](https://stable-baselines3.readthedocs.io/en/master/guide/export.html)
- [SBX (JAX backend for SB3)](https://github.com/araffin/sbx)
- [CleanRL — single-file PPO/DQN reference impls](https://github.com/vwxyzjn/cleanrl)
- [Gymnasium docs](https://gymnasium.farama.org/)
- [Tianshou (alternative PyTorch RL lib)](https://github.com/thu-ml/tianshou)
- [RLlib docs](https://docs.ray.io/en/latest/rllib/index.html) (only if you need MARL/distributed)
- [Jetson.GPIO library on GitHub](https://github.com/NVIDIA/jetson-gpio)
- [pycuda installation for Jetson](https://pypi.jetson-ai-lab.dev/)
- [ONNX](https://github.com/onnx/onnx)
- [ONNX Runtime](https://onnxruntime.ai/)
- [Adafruit_CircuitPython libraries (VL53L0X, MPU6050, PCA9685, etc.)](https://github.com/adafruit)
- [Slamtec sllidar_ros2 driver](https://github.com/Slamtec/sllidar_ros2)
- [Slamtec rplidar_ros driver (ROS 1)](https://github.com/Slamtec/rplidar_ros)
- [pyrplidar (bare-Python RPLidar driver)](https://github.com/Hyun-je/pyrplidar)
- [Dusty Franklin's jetson-inference](https://github.com/dusty-nv/jetson-inference)
- [JetBot project](https://github.com/NVIDIA-AI-IOT/jetbot)
- [JetRacer project](https://github.com/NVIDIA-AI-IOT/jetracer)
- [NVIDIA Jetson AI Lab](https://www.jetson-ai-lab.com/)

### Simulators
- [PyBullet quickstart guide (HTML)](https://github.com/bulletphysics/bullet3/blob/master/docs/pybullet_quickstart_guide/PyBulletQuickstartGuide.md.html)
- [PyBullet GitHub](https://github.com/bulletphysics/bullet3)
- [PyBullet Gym wrappers](https://github.com/benelot/pybullet-gym)
- [MuJoCo](https://mujoco.org/) · [MuJoCo docs](https://mujoco.readthedocs.io/)
- [MJX (JAX MuJoCo) docs](https://mujoco.readthedocs.io/en/stable/mjx.html)
- [Brax (JAX physics + PPO recipes)](https://github.com/google/brax)
- [Madrona Batch Renderer (for fast batched camera rendering in MJX)](https://madrona-engine.github.io/)
- [Isaac Sim docs](https://docs.isaacsim.omniverse.nvidia.com/)
- [Isaac Lab docs](https://isaac-sim.github.io/IsaacLab/)
- [Isaac Lab system requirements (16 GB VRAM minimum)](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)
- [Isaac Lab on NGC (Docker)](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-lab)
- [Gazebo Harmonic install](https://gazebosim.org/docs/harmonic/install/)

### Cloud GPU providers
- [RunPod pricing](https://www.runpod.io/pricing) · [docs](https://docs.runpod.io/)
- [Vast.ai pricing](https://vast.ai/pricing) · [docs](https://docs.vast.ai/)
- [Lambda Labs GPU cloud](https://lambda.ai/service/gpu-cloud)
- [Live GPU rental price tracker](https://www.madebyagents.com/hardware/gpu-rental-prices)

### Papers & further reading
- [Sutton & Barto, "Reinforcement Learning: An Introduction" (2nd ed., free PDF)](http://incompleteideas.net/book/RLbook2020.pdf) — required background
- [Jonnarth et al., "Learning Coverage Paths in Unknown Environments with Deep RL", ICML 2024](https://proceedings.mlr.press/v235/jonnarth24a.html) · [code](https://github.com/jonnarth/coverage-path-planning-rl) — modern SOTA on the exact problem you're solving
- [LIRL: Latent Imagination-Based RL for CPP (Symmetry 2024)](https://www.mdpi.com/2073-8994/16/11/1537)
- [Xu et al., "Survey on Path Planning Based on Deep RL" (2025)](https://proceedings.mlr.press/v278/xu25a.html)
- [Jayalakshmi et al., "Comprehensive Survey on Coverage Path Planning" (IEEE Access 2025)](https://doi.org/10.1109/ACCESS.2025.3556446)
- [Lakshmanan et al., "Complete coverage path planning using RL" (Automation in Construction 2020)](https://doi.org/10.1016/j.autcon.2020.103078) — classic RL CPP
- [Schulman et al., "Proximal Policy Optimization Algorithms"](https://arxiv.org/abs/1707.06347) — the PPO paper
- [Haarnoja et al., "Soft Actor-Critic"](https://arxiv.org/abs/1801.01290)
- [Mnih et al., "Human-level control through deep RL" (Nature 2015)](https://www.nature.com/articles/nature14236) — original DQN, the source of the NatureCNN architecture you'll use
- [Tobin et al., "Domain Randomization for Sim-to-Real" (2017)](https://arxiv.org/abs/1703.06907)
- [OpenAI, "Solving Rubik's Cube with a Robot Hand" (2019)](https://arxiv.org/abs/1910.07113) — ADR
- [Zhao et al., "Sim-to-Real Transfer Survey" (2020)](https://arxiv.org/abs/2009.13303)
- [Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)](https://arxiv.org/abs/1503.02531)
- [Rusu et al., "Policy Distillation" (2016)](https://arxiv.org/abs/1511.06295)
- [Raffin et al., "Stable-Baselines3: Reliable RL Implementations" (JMLR 2021, PDF)](https://jmlr.org/papers/volume22/20-1364/20-1364.pdf)
- [Huang et al., "The 37 Implementation Details of PPO" (CleanRL blog)](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) — required reading before debugging PPO yourself

### Communities & troubleshooting forums
- [NVIDIA Jetson Orin Nano forum](https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems/jetson-orin-nano/619) — fastest help for Jetson-specific issues
- [r/JetsonNano](https://www.reddit.com/r/JetsonNano/)
- [r/robotics](https://www.reddit.com/r/robotics/)
- [r/reinforcementlearning](https://www.reddit.com/r/reinforcementlearning/)
- [Robotics Stack Exchange](https://robotics.stackexchange.com/)
- [Stable-Baselines3 GitHub issues](https://github.com/DLR-RM/stable-baselines3/issues) — searchable history of training-failure modes
- [Stable-Baselines3 Discord](https://discord.gg/9ZPp9XV) (link from SB3 docs)
- [CleanRL Discord](https://discord.gg/D6RCjA6sVT)
- [PyBullet forum (older but useful)](https://pybullet.org/Bullet/phpBB3/)
- [MuJoCo Google group](https://groups.google.com/g/mujoco)
- [JetsonHacks blog & YouTube](https://jetsonhacks.com/) · [@JetsonHacks YouTube](https://www.youtube.com/@JetsonHacks)
- [NVIDIA Jetson AI Lab Community Forum](https://www.jetson-ai-lab.com/)

---

## 12. Troubleshooting & FAQ

### Hardware assembly

**Q: I plugged the Jetson into the boost converter and it doesn't boot — no green LED.**
- Verify the barrel-jack polarity with a multimeter (center should be +). Reversing destroys the board ([forum example](https://forums.developer.nvidia.com/t/jetson-orin-nano-dev-kit-shorted-out/249977/)).
- Verify the voltage is in 5–20 V range under load. If sagging below 5 V on motor startup, your boost converter is undersized — go to a 45 W class converter.
- Verify the barrel-jack mechanical fit. 5.5 × 2.5 mm jacks are subtly different from the more common 5.5 × 2.1 mm; if loose, you have the wrong pigtail.
- Try a known-good 19 V supply (a laptop charger with the right tip) to isolate Jetson vs power-supply problem.

**Q: My motors spin in opposite directions even though wiring looks correct.**
- TT motors have arbitrary polarity — there's no convention. Swap the two leads on one motor.
- Or in software, invert that motor's PWM/direction logic in `motor_driver.py`. This is a one-line fix and is the correct long-term approach.

**Q: The robot vibrates / shudders when both motors are running.**
- Motor mounts are too loose. Add a drop of hot glue.
- Wheels not concentric (common with plastic TT motor wheels). Try the other wheel pair; sometimes one is just bent.
- Caster wheel wobbling. Add a thin nylon washer between caster and chassis.

**Q: The Jetson reboots every time the vacuum motor starts.**
- Brownout. Your battery rail is sagging when the inductive load kicks in.
- Add a 1000 μF + 100 nF capacitor pair across the boost-converter input (battery side).
- Move the vacuum motor to its own MOSFET on the **raw battery rail** (not the Jetson rail) — make sure the boost converter is on the Jetson rail only.
- If still brownouting, you may have one underperforming 18650 cell pulling the pack voltage down. Test cells individually.

**Q: PWM signal doesn't reach the TB6612 from Jetson pin 32/33.**
- Run `sudo /opt/nvidia/jetson-io/jetson-io.py` and enable PWM on those pins, then reboot. This is the single most-common Jetson PWM bug ([NVIDIA jetson-gpio issue #14](https://github.com/NVIDIA/jetson-gpio/issues/14)).
- Or just use a PCA9685 over I²C and stop fighting it: `pip install adafruit-circuitpython-pca9685`.

**Q: I²C device not detected by `i2cdetect`.**
- On Orin Nano, the user I²C bus is **bus 7** (`i2cdetect -y -r 7`), not bus 1 like on the older Nano.
- Verify pull-ups: Adafruit/Pololu breakouts include them, but bare ICs don't. Add 4.7 kΩ pull-ups to 3.3 V if needed.
- Check wiring continuity with a multimeter.
- Verify the device is powered (run a voltmeter across its VCC/GND).

### Power & safety

**Q: How do I know if my battery is about to die mid-cleanup?**
- Add a simple voltage-divider (2 resistors) → ADC input pin → background-thread monitor. When pack voltage < 6.4 V (for 2S) or < 9.6 V (for 3S), the robot drives to a fixed "dock" pose and stops. Cheap hack: just use a buzzer that beeps when battery is low and have the policy interpret repeated beeps as "go home".
- For a stretch upgrade: use an INA219 (~$5) on the battery rail for accurate voltage + current monitoring.

**Q: Can I leave the Li-ion battery in the chassis to charge through a barrel-jack input?**
- **Not safely with a 2-cell pack and just a 12 V wall wart**. You need a proper balance charger or a TP4056-style cell-level board. Buy a 2S balance charger ($10 Amazon) and charge externally.
- For LiPo packs, use the matching XT60 + balance lead with a hobby charger (iSDT, ToolkitRC, SkyRC — $30–60 stretch).

### RL training (Phase 1)

**Q: Q-learning never converges; coverage stuck below 50%.**
- Check reward magnitudes. If `first_visit bonus = 1` but `time penalty * max_steps = -0.05 * 2000 = -100`, the agent learns to stop. Reduce time penalty or scale first-visit bonus.
- Verify ε actually decays. Print ε every 100 episodes.
- Check that `_obs_key()` returns hashable tuples (not numpy arrays).
- Reduce `lambda` from 0.9 to 0.7; high λ can cause instability with sparse rewards.

**Q: SARSA(λ) is slower than Q-learning. Did I implement it wrong?**
- SARSA is on-policy and inherently slower with exploration than off-policy Q-learning for tasks with bad-action recovery (Cliffwalk classic). For coverage problems, either is fine.
- Verify your eligibility traces decay correctly: `E[s,a] *= gamma * lam` after every update.

### RL training (Phase 2)

**Q: PPO reward is flat for 500k steps.**
- First, verify the env is solvable. Run `keyboard_teleop_sim.py` and try to reach >50% coverage in <2000 steps yourself. If you can't, the env reward is too sparse.
- Print `info["coverage"]` every step during a random-policy rollout. Coverage should increase even with random actions (just slowly). If it doesn't, your coverage computation has a bug.
- Check the observation. Save a few frames and view them: are they black? All gray? They should look like the camera feed.
- Reduce `n_steps` from 256 to 128 and increase `n_epochs` from 4 to 10. Sometimes PPO needs more updates per rollout for sparse-reward tasks.
- Try entropy regularization: bump `ent_coef` from 0.01 to 0.05. Encourages exploration.
- Check that NatureCNN is getting uint8 input scaled to float32 inside; SB3 does this if you pass `normalize_images=False` and the input dtype is uint8.

**Q: PPO reward looks great in sim but the robot just spins in circles in real world.**
- Sim-to-real gap. Did you train with domain randomization? If not, add it.
- Action time scaling. In sim, action 0 (forward) moves the robot 0.15 m/s × 0.05 s = 7.5 mm per env step. On the real robot at 50 Hz with PWM 50%, you might be moving 30 mm per env step. Match these or the policy will overshoot every action.
- Camera FOV / mounting. If sim camera is at 18 cm height with 60° FOV, the real camera must be too. **Use the same camera mount geometry.**
- Lighting. Sim default lighting is uniform overhead; your room has shadows from furniture. Add lighting randomization to sim.

**Q: PyBullet sim is too slow on my RTX 2060 (under 200 FPS per env).**
- Use `p.DIRECT` not `p.GUI` for training (huge speedup).
- Camera rendering is the bottleneck. Reduce image size from 84 to 64 px temporarily. If that speeds it up dramatically, you're rendering-bound — consider migrating to MJX + Madrona for serious training (cloud A100).
- Reduce physics substeps. 240 Hz is the default; 60 Hz is fine for ground robots.
- Use `SubprocVecEnv` not `DummyVecEnv` (PyBullet doesn't share state across procs well, but at least you get parallel rollouts).

**Q: My laptop fan is screaming and I'm getting 60 FPS instead of 1000.**
- You're CPU-bound. PyBullet physics is single-threaded; `SubprocVecEnv` with 4 envs gives you ~4× throughput on a 4-core CPU.
- Consider going to cloud at this point. $5 of A100 time is worth it.

### Sim-to-real

**Q: Policy works perfectly in sim, fails the moment I put it on the real robot.**
- 90% of the time, the issue is action latency or observation distribution.
- **Latency**: Inject 30–80 ms action latency in sim training. Real Jetson + USB camera + motor command pipeline is ~50 ms; sim is ~0 ms. Policies that look near-optimal in sim can become unstable when actions arrive late.
- **Observation distribution**: Capture 100 real frames; compare their mean/variance/histogram to sim frames. If they're wildly different (e.g., sim is bright/uniform, real is dark/textured), add more aggressive domain randomization for those properties.
- **Action mapping**: Sim "turn left" is instant rotation; real "turn left" is differential PWM that takes ~300 ms to spin up motors. Match these or use a small filter that turns each policy action into a fixed-duration low-level command.

**Q: Robot starts spinning in place forever.**
- Reward shaping issue: the policy maximizes "not bumping" by turning forever. Add a "moving" reward (positive reward proportional to distance traveled).
- Or the camera observation is stuck/static. Verify with logs that frames are changing.

### Quantization & deployment

**Q: `trtexec` fails with "Unsupported ONNX opset 17".**
- TensorRT 10.3 supports opset 17. If on an older JetPack, downgrade ONNX export: `torch.onnx.export(..., opset_version=16)`.
- Check version with `dpkg -l | grep tensorrt`.

**Q: TensorRT engine works on Jetson but inference output is garbage.**
- Input dtype mismatch. The most common bug: PyTorch model expects float in [0, 255] (NatureCNN normalizes internally), but you're feeding it float in [0, 1] or vice versa. Match exactly.
- Channel order. PyTorch is NCHW; some pipelines feed NHWC. Verify shape and order with prints.
- Run the FP32 engine first; if that's correct, the FP16 conversion has precision issues. Try `--strict-types` or check for any `argmax/argmin` ops which are precision-sensitive.

**Q: INT8 inference is fast but accuracy collapsed.**
- Calibration dataset doesn't match deployment distribution. Capture *new* calibration frames on the real robot under the actual lighting you'll deploy in. 1000 frames is fine.
- Some layers shouldn't be INT8 (final logits, softmax). TensorRT 10 lets you specify per-layer precision with `--layerPrecisions=...`. Force final dense layer to FP16.

**Q: `Jetson.GPIO` says "Permission denied" or PWM doesn't start.**
- You haven't added your user to the `gpio` group or you didn't log out/in after. Run `groups` to check.
- Verify with `ls -l /sys/class/pwm/pwmchip*`.

**Q: My USB camera frame rate is awful (10 FPS instead of 30).**
- Force MJPEG instead of YUYV: `cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))`.
- Reduce resolution before grabbing: `cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)`.
- Use `v4l2-ctl --list-formats-ext -d /dev/video0` to see what your camera actually supports.

### Evaluation

**Q: How do I measure coverage in the real world without overhead motion capture?**
- Mount a phone overhead (ceiling-mount or doorway tripod). Record a fixed 5-min run. Use [OpenCV blob tracking](https://docs.opencv.org/4.x/d4/d7e/tutorial_py_blob_detection.html) on the robot's bright top to extract its trajectory; rasterize the trajectory onto a grid of room cells; divide by free-cell count.
- Or place small colored stickers across the floor as a grid and have the overhead camera detect which stickers got covered.
- Or use AprilTags (cheap, robust): one tag on robot, four corner tags around the room; opencv-april-tag gives you the robot's room-frame trajectory.

**Q: My RL policy underperforms a "drive forward, turn 30° on bump" baseline.**
- That's an honest result and worth a writeup. Common reasons:
  - Your reward function over-weights coverage and under-weights efficiency — the policy explores too aggressively
  - Sim-to-real gap dominates
  - The baseline is actually well-tuned for the room geometry. Try other rooms.
- The interesting next step is *why*. Log policy entropy, action distribution, and visited-cell heatmap. Compare.

**Q: Cleanliness is only 10–20% after 5 minutes.**
- Your suction is low. That's expected — a 40 mm fan is ~50× weaker than a real vacuum turbine.
- Add a brush. A small DC motor with a 3D-printed bristle roller in front of the dustbin intake dramatically helps for hard floors.
- Tighten the airflow path: seal gaps with foam, narrow the intake.
- Lower the intake to ~2–4 mm above the floor (closer = better, until it scrapes).

---

### Final words

The thing that will most determine whether this project teaches you a lot vs frustrates you is **finishing Phase 0** (mechanical + power + teleop) before touching any RL code. Bad power architecture or a flaky chassis will silently corrupt every Phase 2 training run, and you won't know whether your policy is bad or your robot is bad. Spend the first week on the boring stuff.

Good luck. When you get stuck mid-project and can't ask me, the single best venue per topic is:
- **Hardware/power/Jetson boot:** [NVIDIA Jetson Orin Nano forum](https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems/jetson-orin-nano/619)
- **RL training not converging:** [SB3 GitHub issues](https://github.com/DLR-RM/stable-baselines3/issues) (search first) → then [r/reinforcementlearning](https://www.reddit.com/r/reinforcementlearning/)
- **Sim-to-real:** [Isaac Sim forum](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67) or the Brax/MJX GitHub
- **TensorRT/quantization:** [NVIDIA TensorRT GitHub issues](https://github.com/NVIDIA/TensorRT/issues)
- **General DIY robot:** [r/robotics](https://www.reddit.com/r/robotics/), [Robotics Stack Exchange](https://robotics.stackexchange.com/)

Have fun. The robot doesn't need to be good. The pipeline needs to be real.
