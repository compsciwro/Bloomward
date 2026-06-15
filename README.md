<h1 align="center">🌸 Bloomward</h1>

<p align="center">
  <img src="https://github.com/compsciwro/Bloomward/blob/main/assets/logo_img.png?raw=true" alt="Bloomward Logo" width="500"/>
</p>

<p align="center">
  <i>A competitive hex based restoration strategy game designed for reinforcement learning ✨</i>
</p>

---

## 🌼 About the Project

**Bloomward** is a custom strategy game project built around **flowers, spirits, seasons, and spreading corruption**.  
It is being designed as both:

🌱 a fun and strategic game for human play  
🤖 a structured environment for reinforcement learning experiments  

Players compete on a shared hex board, placing flowers, forming combos, activating spirits, and trying to survive the pressure of corruption as it spreads inward toward the Sacred Tree.

---

## 🎯 Main Idea

Bloomward explores how a game can be:

- 🌸 strategically interesting for players
- 🧠 suitable for AI and reinforcement learning
- 🍃 themed around restoration, balance, and environmental pressure

Instead of using a standard benchmark game, this project creates a **custom game environment from scratch**.

---

## 🕹️ Core Gameplay

### 🌳 Board Structure
The board has three main layers:

- **Sacred Core** 🌳  
  The center of the board, containing the Sacred Tree

- **Fertile Soil** 🌱  
  The main playable area where flowers can be placed

- **Corrupt Soil** 🖤  
  The outer edge, where corruption begins and spreads inward

---

### 🌷 Flower System
Current flower types include:

- 🌻 Sunflower
- 🌷 Tulip
- 🌼 Blossom

Players draw and place flowers on valid fertile tiles.

---

### ✨ Combos and Spirits
Flower placements can form combos that awaken spirits.

- 🔺 **3 matching flowers in a triangle** activates a spirit
- 🌸 **9 flower cluster** activates a stronger restoration effect

Spirits help defend, heal, or stabilize the board.

---

### 🖤 Corruption System
Corruption is one of the main pressure mechanics in the game.

- starts at the outer edge
- spreads inward over time
- only spreads onto empty fertile soil
- reduces space and threatens the Sacred Tree

This creates urgency and forces smarter decisions.

---

### 🍂 Seasons
Seasonal changes keep the game dynamic and can affect how pressure builds across the board.

---

## 🤖 Reinforcement Learning Goal

Bloomward is also being developed as a reinforcement learning environment.

The long term goal is to model the game so that an agent can learn through interaction with the board by:

- observing game states
- choosing valid actions
- receiving rewards
- adapting to corruption, seasons, and an opponent

This makes Bloomward a useful testbed for **game AI and sequential decision making**.

---

## 🛠️ Current Project Goals

This project currently focuses on:

- 📘 formalising the game rules
- 🎮 building a digital prototype
- 🧩 modelling the environment for reinforcement learning
- 🤖 preparing for future agent training

---

## 📌 Prototype Scope

The first prototype includes:

- hexagonal board representation
- two player turn system
- flower placement
- combo detection
- spirit activation
- corruption spread
- season cycling
- win and loss checking

For now, the priority is **functionality over polish**.

---

## 🌟 Future Plans

Planned next steps include:

- improving the prototype
- refining rule consistency
- defining state and action spaces
- designing reward functions
- testing baseline agents
- training reinforcement learning agents

---

## 👤 Author

**R. Solomons**  
🎓 Honours Project 2026  
🏫 University of the Western Cape  

---

## 💖 Notes

Bloomward is currently being developed for academic and research purposes.  
A formal license and fuller documentation can be added later.

---

<p align="center">
  🌸 Bloomward • Strategy • Restoration • Reinforcement Learning 🤖
</p>
