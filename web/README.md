# Bucket Brigade Visualizer

A modern web application for visualizing multi-agent cooperation in firefighting scenarios.

## Features

- 🎮 **Game Replay**: Step-by-step visualization of firefighting games
- 🏆 **Agent Rankings**: Performance tracking and leaderboard
- 📊 **Interactive Dashboard**: Overview of experiments and results
- 💾 **Session Storage**: Client-side data management
- 🎨 **Modern UI**: Built with React, TypeScript, and Tailwind CSS

## Tech Stack

- **Framework**: React 18 with TypeScript
- **Build Tool**: Vite
- **Styling**: Tailwind CSS with custom theme
- **Icons**: Lucide React
- **Charts**: Recharts (planned)
- **Data**: Session Storage (upgradeable to database)

## Development

### Prerequisites

- Node.js 18+
- npm or yarn

### Setup

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

### Project Structure

```
web/
├── src/
│   ├── components/     # Reusable UI components
│   │   ├── GameBoard.tsx      # Firefighting visualization
│   │   ├── ReplayControls.tsx # Playback controls
│   │   └── GameInfo.tsx       # Scenario and stats display
│   ├── pages/          # Main application pages
│   │   ├── Dashboard.tsx      # Overview and stats
│   │   ├── GameReplay.tsx     # Game visualization
│   │   ├── Rankings.tsx       # Agent performance
│   │   └── Settings.tsx       # Data management
│   ├── types/          # TypeScript type definitions
│   ├── utils/          # Utility functions
│   ├── App.tsx         # Main application component
│   └── main.tsx        # Application entry point
├── public/             # Static assets
└── package.json        # Dependencies and scripts
```

## Usage

### Loading Data

1. **Run Experiments**: Use the Python scripts to generate game data
   ```bash
   python scripts/run_batch.py --num-games 50
   ```

2. **Upload Results**: Use the Settings page to import:
   - Game replays (JSON files from `replays/` directory)
   - Batch results (CSV files from `results/` directory)

3. **Explore**: Navigate between Dashboard, Game Replay, and Rankings

### Features

#### Game Replay
- Interactive ring-based game board
- Step-by-step night progression
- Agent position and action visualization
- Scenario parameter display
- Real-time statistics

#### Rankings
- Agent performance leaderboard
- Statistical analysis of game results
- Uncertainty quantification
- Historical performance tracking

#### Dashboard
- Experiment overview and statistics
- Quick access to recent games
- Getting started guide

## Data Management

The application uses session storage for data persistence:

- **Game Replays**: Stored as `STORAGE_KEYS.GAME_REPLAYS`
- **Batch Results**: Stored as `STORAGE_KEYS.BATCH_RESULTS`
- **Settings**: Stored as `STORAGE_KEYS.UI_SETTINGS`

Data can be exported/imported via the Settings page.

## Future Enhancements

- Database integration for persistent storage
- Real-time experiment monitoring
- Advanced ranking visualizations
- Agent strategy analysis
- Comparative performance charts
- Export capabilities (videos, reports)

## Contributing

1. Follow the existing TypeScript and React patterns
2. Use Tailwind CSS for styling
3. Maintain type safety throughout
4. Test components with realistic data
5. Follow the established file structure

## License

See main repository LICENSE file.
