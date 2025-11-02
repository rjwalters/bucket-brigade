BUCKET BRIGADE — VISUALIZER DESIGN
-----------------------------------

PURPOSE
--------
The visualizer presents saved Bucket Brigade games in an interactive 2D display.
It is implemented using HTML, CSS, and TypeScript (with optional React or plain
DOM rendering). The goal is to make each night's actions and outcomes easy to
understand and to produce visually appealing replays from recorded JSON files.

PRIMARY OBJECTIVES
------------------
1. Represent the ten-house ring clearly and consistently.
2. Animate state changes: fires, extinguishing, and ruin transitions.
3. Display agent positions, signals, and actions intuitively.
4. Allow playback control, night-by-night stepping, and speed adjustment.
5. Integrate easily with replay JSON files exported from the simulator.

DATA INPUT
-----------
The visualizer consumes JSON replay files produced by the environment:

  {
    "scenario": { "beta": 0.25, "kappa": 0.5, "num_agents": 6, ... },
    "nights": [
      {
        "night": 1,
        "houses": [0,1,0,0,2,0,0,0,0,0],
        "signals": [1,0,1,1,0,1],
        "actions": [[3,1],[4,0],[2,1],[3,1],[5,0],[7,1]],
        "rewards": [0.2,0.1,0.2,0.2,0.1,0.3]
      },
      ...
    ]
  }

Houses are encoded as integers: 0=Safe, 1=Burning, 2=Ruined.
Signals and actions are encoded as integers: 0=REST, 1=WORK.

CSS VISUALIZATION CONCEPT
-------------------------
The display shows a circular town with 10 equally spaced houses.
Each house is drawn as a div positioned on a ring using CSS transforms.

HTML Structure (simplified):

  <div id="town">
    <div class="house safe" data-index="0"></div>
    <div class="house burning" data-index="1"></div>
    ...
    <div class="agent" data-agent="0"></div>
    <div class="agent" data-agent="1"></div>
    ...
  </div>

The ring layout uses absolute positioning and transform-rotate for
equal spacing around a central point.

  .house {
    position: absolute;
    width: 60px;
    height: 60px;
    border-radius: 50%;
    transform-origin: center center;
    left: 50%; top: 50%;
  }

  .safe     { background-color: #4caf50; }
  .burning  { background-color: #e53935; animation: flicker 0.3s infinite; }
  .ruined   { background-color: #424242; }

  @keyframes flicker {
    0% { filter: brightness(1); }
    50% { filter: brightness(1.5); }
    100% { filter: brightness(1); }
  }

Agents are shown as smaller dots or icons orbiting the ring at the
position of their chosen house. Signals are indicated by color or icon
(e.g., a small badge on the agent).

  .agent {
    position: absolute;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    border: 2px solid white;
    transform-origin: center center;
  }

  .signal-work::after {
    content: "🔥";
    position: absolute;
    top: -12px;
    left: 4px;
  }

  .signal-rest::after {
    content: "💤";
    position: absolute;
    top: -12px;
    left: 4px;
  }

Night-by-night transitions are implemented using CSS transitions
or JavaScript animations to fade colors or move agents smoothly.

TIMELINE AND CONTROLS
---------------------
Controls appear beneath the town visualization.

  [⏮️] [⏯️] [⏭️] [Speed ▾]
  Night: 7 / 18

Implemented using TypeScript event handlers:

  - Play/pause toggles automatic stepping through nights.
  - Forward/back moves one night at a time.
  - Speed adjusts frame delay between transitions.
  - Display updates agent signals, positions, and house states.

SCENARIO AND SUMMARY PANEL
--------------------------
A sidebar (fixed div) displays scenario parameters and summary stats.

  SCENARIO
  beta: 0.25
  kappa: 0.5
  p_spark: 0.02

  TEAM RESULTS
  Nights: 18
  Houses saved: 7
  Houses ruined: 3
  Team reward: 241.6

  INDIVIDUAL REWARDS
  Agent 0: 32.5
  Agent 1: 28.0
  ...

This panel updates automatically when a new replay file is loaded.

INTERACTIVITY
-------------
- Hovering over a house shows which agents worked there that night.
- Hovering over an agent shows its honesty record and total contribution.
- Clicking "auto-play" starts a continuous loop of nights.
- Replays can be paused, reversed, or sped up.
- Optionally display honesty flags (e.g., outline agent in yellow if lying).

CSS AND VISUAL STYLE
--------------------
- Use neutral backgrounds and high-contrast color for fire effects.
- Color palette:
    Safe:    #4caf50 (green)
    Burning: #e53935 (red)
    Ruined:  #424242 (gray)
    Agent:   #2196f3 (blue)
- Soft drop shadows and rounded corners for a polished look.
- Use CSS transitions (0.3–0.5s) for smooth updates.

FILE STRUCTURE
--------------
web/
├── public/
│   └── index.html
├── src/
│   ├── index.ts
│   ├── replayLoader.ts
│   ├── visualize.ts
│   ├── components/
│   │   ├── Town.ts
│   │   ├── AgentLayer.ts
│   │   ├── Controls.ts
│   │   └── Sidebar.ts
│   └── styles/
│       ├── town.css
│       └── controls.css

REPLAY LOADING
--------------
Replays are loaded from local file input or a URL.

Example TypeScript:
  const input = document.getElementById('replayFile');
  input.onchange = async e => {
      const file = (e.target as HTMLInputElement).files[0];
      const text = await file.text();
      const replay = JSON.parse(text);
      startVisualization(replay);
  };

Each frame update uses replay.nights[t] to update CSS classes for
houses and agent elements.

SCALABILITY
-----------
The CSS visualizer is lightweight enough to run hundreds of replays in
the browser with no WebGL dependency. For later upgrades:
  - Switch to Canvas or WebGL for continuous animation.
  - Add networked dashboard to fetch and replay recent simulations.
  - Allow side-by-side comparison of different games.

SUMMARY
-------
The CSS visualizer renders the ten-house town in a circular layout with
simple color-coded transitions for fire dynamics. Agent icons display
signals and positions. Replays are loaded from JSON, stepped through by
JavaScript, and styled with pure CSS animations. This approach provides
an interpretable, low-dependency visualization tool suitable for public
demos, analysis, and educational use.
