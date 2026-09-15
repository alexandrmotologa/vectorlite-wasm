const fs = require('fs');
const path = require('path');

function buildLogoSvg() {
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024" width="1024" height="1024">
  <defs>
    <clipPath id="squircle-clip">
      <rect x="24" y="24" width="976" height="976" rx="220" />
    </clipPath>

    <!-- Gradients -->
    <linearGradient id="cyan-glow" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#00f5ff"/>
      <stop offset="100%" stop-color="#0284c7"/>
    </linearGradient>

    <linearGradient id="amber-accent" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#f59e0b"/>
      <stop offset="100%" stop-color="#b45309"/>
    </linearGradient>

    <filter id="subtle-shadow" x="-10%" y="-10%" width="120%" height="120%">
      <feDropShadow dx="0" dy="18" stdDeviation="22" flood-color="#0f172a" flood-opacity="0.22" />
    </filter>
  </defs>

  <!-- Luxury White Squircle Container -->
  <rect x="24" y="24" width="976" height="976" rx="220" fill="#ffffff" stroke="#e2e8f0" stroke-width="6" />

  <g clip-path="url(#squircle-clip)">
    <g transform="translate(512, 512)" filter="url(#subtle-shadow)">

      <!-- Hexagonal Architectural Gateway Frame -->
      <polygon points="
        0,-375
        325,-187
        325,187
        0,375
        -325,187
        -325,-187
      " fill="none" stroke="#0f172a" stroke-width="32" stroke-linejoin="round" />

      <!-- Inner Dashed Telemetry Gateway -->
      <polygon points="
        0,-345
        298,-172
        298,172
        0,345
        -298,172
        -298,-172
      " fill="none" stroke="#00f5ff" stroke-width="3.5" opacity="0.5" stroke-dasharray="12, 10" />

      <!-- Spatial Compass / Radar Rings -->
      <circle cx="0" cy="0" r="260" fill="none" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="6, 8" opacity="0.3" />
      <line x1="0" y1="-320" x2="0" y2="320" stroke="#cbd5e1" stroke-width="1.5" stroke-dasharray="6, 8" opacity="0.35" />
      <line x1="-275" y1="0" x2="275" y2="0" stroke="#cbd5e1" stroke-width="1.5" stroke-dasharray="6, 8" opacity="0.35" />

      <!-- ================================================================ -->
      <!-- THE VECTOR PEREGRINE RAPTOR (Apex Predator of Vector Space)      -->
      <!-- 100% Watertight Solid Base Silhouette                           -->
      <!-- ================================================================ -->
      <path d="
        M 0 -310
        L 60 -275 L 120 -230 L 175 -135 L 230 -15 L 265 125 L 245 235 L 180 295 L 100 330 L 0 345
        L -100 330 L -180 295 L -245 235 L -265 125 L -230 -15 L -175 -135 L -120 -230 L -60 -275 Z
      " fill="#0b0f19" />

      <!-- Crown & Aerodynamic Brow Crest -->
      <polygon points="0,-310 60,-275 30,-220 0,-240" fill="#475569" />
      <polygon points="0,-310 -60,-275 -30,-220 0,-240" fill="#334155" />
      <polygon points="60,-275 120,-230 75,-175 30,-220" fill="#334155" />
      <polygon points="-60,-275 -120,-230 -75,-175 -30,-220" fill="#1e293b" />

      <!-- Swept Geometric Stealth Wing Facets (Right) -->
      <polygon points="120,-230 175,-135 125,-100 75,-175" fill="#1e293b" />
      <polygon points="175,-135 230,-15 165,0 125,-100" fill="#475569" />
      <polygon points="230,-15 265,125 195,125 165,0" fill="#334155" />
      <polygon points="265,125 245,235 175,205 195,125" fill="#1e293b" />
      <polygon points="245,235 180,295 125,250 175,205" fill="#475569" />
      <polygon points="180,295 100,330 65,270 125,250" fill="#334155" />
      <polygon points="100,330 0,345 0,285 65,270" fill="#0f172a" />

      <!-- Swept Geometric Stealth Wing Facets (Left Symmetrical) -->
      <polygon points="-120,-230 -175,-135 -125,-100 -75,-175" fill="#0f172a" />
      <polygon points="-175,-135 -230,-15 -165,0 -125,-100" fill="#334155" />
      <polygon points="-230,-15 -265,125 -195,125 -165,0" fill="#1e293b" />
      <polygon points="-265,125 -245,235 -175,205 -195,125" fill="#475569" />
      <polygon points="-245,235 -180,295 -125,250 -175,205" fill="#334155" />
      <polygon points="-180,295 -100,330 -65,270 -125,250" fill="#1e293b" />
      <polygon points="-100,330 0,345 0,285 -65,270" fill="#0f172a" />

      <!-- Forehead Center Diamond -->
      <polygon points="0,-240 30,-220 0,-155 -30,-220" fill="#1e293b" />

      <!-- Fierce Raptor Eyebrows (Sloping down to center: intense predator gaze) -->
      <polygon points="0,-155 30,-165 75,-175 45,-140 0,-130" fill="#0f172a" stroke="#00f5ff" stroke-width="1" />
      <polygon points="0,-155 -30,-165 -75,-175 -45,-140 0,-130" fill="#1e293b" stroke="#00f5ff" stroke-width="1" />

      <!-- Deep Recessed Eye Sockets (Obsidian) -->
      <polygon points="20,-130 70,-155 60,-125 18,-115" fill="#0b0f19" />
      <polygon points="-20,-130 -70,-155 -60,-125 -18,-115" fill="#0b0f19" />

      <!-- Electric Cyan Optics (Sharp Predatory Slit, Canting UPWARD to ears: (22,-126) to (65,-150)!) -->
      <polygon points="22,-126 65,-150 56,-130 20,-118" fill="url(#cyan-glow)" />
      <polygon points="-22,-126 -65,-150 -56,-130 -20,-118" fill="url(#cyan-glow)" />
      <!-- White Laser Pupil Core -->
      <polygon points="35,-130 52,-142 46,-132 32,-124" fill="#ffffff" />
      <polygon points="-35,-130 -52,-142 -46,-132 -32,-124" fill="#ffffff" />

      <!-- Malar Cheek Facets (Peregrine Falcon Black Helmet / Hood Markings) -->
      <polygon points="45,-140 75,-175 125,-100 65,-80" fill="#0f172a" />
      <polygon points="-45,-140 -75,-175 -125,-100 -65,-80" fill="#0b0f19" />
      <polygon points="65,-80 125,-100 165,0 90,-10" fill="#334155" />
      <polygon points="-65,-80 -125,-100 -165,0 -90,-10" fill="#1e293b" />

      <!-- Muzzle / Upper Cere (Titanium Slate) -->
      <polygon points="0,-130 20,-105 0,-75" fill="#64748b" />
      <polygon points="0,-130 -20,-105 0,-75" fill="#475569" />

      <!-- Compact Hooked Raptor Beak (Charcoal Armor with Amber Cutting Edge) -->
      <polygon points="0,-75 22,-75 16,-25 0,-5" fill="#334155" stroke="#475569" stroke-width="1" />
      <polygon points="0,-75 -22,-75 -16,-25 0,-5" fill="#1e293b" stroke="#334155" stroke-width="1" />
      <!-- Sharp Curved Talon Hook Tip with Gold Cutting Ridge -->
      <polygon points="0,-5 12,-18 0,15" fill="url(#amber-accent)" />
      <polygon points="0,-5 -12,-18 0,15" fill="#92400e" />
      <line x1="0" y1="-75" x2="0" y2="15" stroke="#fde68a" stroke-width="1.5" opacity="0.9" />

      <!-- Throat & Neck Vector Chevron Plates -->
      <polygon points="0,15 35,45 0,80" fill="#1e293b" />
      <polygon points="0,15 -35,45 0,80" fill="#0f172a" />
      <polygon points="35,45 90,-10 0,-5" fill="#334155" />
      <polygon points="-35,45 -90,-10 0,-5" fill="#1e293b" />
      <polygon points="35,45 80,105 0,140" fill="#475569" />
      <polygon points="-35,45 -80,105 0,140" fill="#1e293b" />

      <!-- Central Radiant HNSW Vector Node (384-D Compass Core) -->
      <circle cx="0" cy="180" r="48" fill="#0b0f19" stroke="#00f5ff" stroke-width="4" />
      <circle cx="0" cy="180" r="34" fill="none" stroke="#38bdf8" stroke-width="2" stroke-dasharray="6, 5" />
      <!-- Concentric Diamond Vector Coordinates -->
      <polygon points="0,140 38,180 0,220 -38,180" fill="url(#cyan-glow)" />
      <polygon points="0,152 26,180 0,208 -26,180" fill="#ffffff" opacity="0.9" />

      <!-- Lower Keel Plates -->
      <polygon points="0,228 45,260 0,285" fill="#1e293b" />
      <polygon points="0,228 -45,260 0,285" fill="#0f172a" />

    </g>
  </g>
</svg>
`;
}

const svg = buildLogoSvg();
const svgPath = path.resolve(__dirname, '../docs/images/logo.svg');
fs.writeFileSync(svgPath, svg, 'utf-8');
console.log('✓ Wrote perfected Peregrine Vector logo.svg');
