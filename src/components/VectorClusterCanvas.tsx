import React, { useRef, useEffect, useState, useMemo } from 'react';
import { FastPCA, Point2D, Point3D } from '../engine/pca';
import { TextChunk } from '../engine/chunker';
import { HybridSearchResult } from '../engine/hybrid';
import { Eye, RefreshCw, ZoomIn, ZoomOut, Box, Orbit } from 'lucide-react';

interface VectorClusterCanvasProps {
  vectors: Array<{ id: string; vector: Float32Array }>;
  chunksMap: Map<string, TextChunk>;
  queryVector: Float32Array | null;
  searchResults: HybridSearchResult[];
  onSelectChunk?: (chunk: TextChunk) => void;
}

interface ProjectedPoint {
  id: string;
  x: number;
  y: number;
  z?: number;
  screenX?: number;
  screenY?: number;
  depth?: number;
  docName: string;
  chunk: TextChunk;
}

const DOC_COLORS = [
  '#22c55e', // brand emerald
  '#6366f1', // indigo
  '#06b6d4', // cyan
  '#f59e0b', // amber
  '#ec4899', // pink
  '#8b5cf6', // purple
  '#10b981', // green
];

export const VectorClusterCanvas: React.FC<VectorClusterCanvasProps> = ({
  vectors,
  chunksMap,
  queryVector,
  searchResults,
  onSelectChunk,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [viewMode, setViewMode] = useState<'2D' | '3D'>('2D');
  const [hoveredPoint, setHoveredPoint] = useState<{ point: ProjectedPoint; screenX: number; screenY: number } | null>(null);
  const [zoom, setZoom] = useState(1.0);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [rotation, setRotation] = useState({ yaw: 0.4, pitch: 0.2 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  const docColorMap = useMemo(() => {
    const map = new Map<string, string>();
    let colorIdx = 0;
    for (const v of vectors) {
      const chunk = chunksMap.get(v.id);
      const docName = chunk?.documentName || 'unknown';
      if (!map.has(docName)) {
        map.set(docName, DOC_COLORS[colorIdx % DOC_COLORS.length]);
        colorIdx++;
      }
    }
    return map;
  }, [vectors, chunksMap]);

  // Compute PCA model and project vectors
  const { pcaModel, projectedPoints2D, projectedPoints3D } = useMemo(() => {
    if (vectors.length < 2) {
      return { pcaModel: null, projectedPoints2D: [], projectedPoints3D: [] };
    }

    try {
      const rawVecs = vectors.map((v) => v.vector);
      const model = FastPCA.fit(rawVecs, 15);

      const points2D: ProjectedPoint[] = vectors.map((v) => {
        const coords = FastPCA.project(v.vector, model);
        const chunk = chunksMap.get(v.id)!;
        return {
          id: v.id,
          x: coords.x,
          y: coords.y,
          docName: chunk?.documentName || 'unknown',
          chunk,
        };
      });

      const points3D: ProjectedPoint[] = vectors.map((v) => {
        const coords = FastPCA.project3D(v.vector, model);
        const chunk = chunksMap.get(v.id)!;
        return {
          id: v.id,
          x: coords.x,
          y: coords.y,
          z: coords.z,
          docName: chunk?.documentName || 'unknown',
          chunk,
        };
      });

      return { pcaModel: model, projectedPoints2D: points2D, projectedPoints3D: points3D };
    } catch {
      return { pcaModel: null, projectedPoints2D: [], projectedPoints3D: [] };
    }
  }, [vectors, chunksMap]);

  // Project query vector
  const projectedQuery2D: Point2D | null = useMemo(() => {
    if (!queryVector || !pcaModel) return null;
    return FastPCA.project(queryVector, pcaModel);
  }, [queryVector, pcaModel]);

  const projectedQuery3D: Point3D | null = useMemo(() => {
    if (!queryVector || !pcaModel) return null;
    return FastPCA.project3D(queryVector, pcaModel);
  }, [queryVector, pcaModel]);

  const topResultIds = useMemo(() => {
    return new Set(searchResults.map((r) => r.id));
  }, [searchResults]);

  // Render loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const padding = 40;
    const plotWidth = width - padding * 2;
    const plotHeight = height - padding * 2;

    ctx.clearRect(0, 0, width, height);

    // Background
    ctx.fillStyle = '#050914';
    ctx.fillRect(0, 0, width, height);

    // Grid
    ctx.strokeStyle = 'rgba(30, 41, 59, 0.4)';
    ctx.lineWidth = 1;
    const gridSize = 40 * zoom;
    const offsetX = (pan.x % gridSize);
    const offsetY = (pan.y % gridSize);

    for (let x = offsetX; x < width; x += gridSize) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = offsetY; y < height; y += gridSize) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    if (projectedPoints2D.length === 0) {
      ctx.fillStyle = '#64748b';
      ctx.font = '13px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Load documents above to project embeddings into vector space', width / 2, height / 2);
      return;
    }

    if (viewMode === '2D') {
      // 2D Projection
      const toCanvasCoords = (ptX: number, ptY: number) => {
        const cx = width / 2 + pan.x;
        const cy = height / 2 + pan.y;
        const x = cx + (ptX - 0.5) * plotWidth * zoom;
        const y = cy + (ptY - 0.5) * plotHeight * zoom;
        return { x, y };
      };

      // Radar rings & connecting lines
      if (projectedQuery2D) {
        const qPos = toCanvasCoords(projectedQuery2D.x, projectedQuery2D.y);

        ctx.strokeStyle = 'rgba(34, 197, 94, 0.15)';
        ctx.lineWidth = 1.5;
        [40, 80, 130, 190].forEach((r) => {
          ctx.beginPath();
          ctx.arc(qPos.x, qPos.y, r * zoom, 0, Math.PI * 2);
          ctx.stroke();
        });

        for (const pt of projectedPoints2D) {
          if (topResultIds.has(pt.id)) {
            const ptPos = toCanvasCoords(pt.x, pt.y);
            ctx.beginPath();
            ctx.moveTo(qPos.x, qPos.y);
            ctx.lineTo(ptPos.x, ptPos.y);
            ctx.strokeStyle = 'rgba(74, 222, 128, 0.4)';
            ctx.lineWidth = 1.5;
            ctx.setLineDash([4, 4]);
            ctx.stroke();
            ctx.setLineDash([]);
          }
        }
      }

      // Render points
      for (const pt of projectedPoints2D) {
        const pos = toCanvasCoords(pt.x, pt.y);
        pt.screenX = pos.x;
        pt.screenY = pos.y;
        const isTopMatch = topResultIds.has(pt.id);
        const color = docColorMap.get(pt.docName) || '#22c55e';

        if (isTopMatch) {
          ctx.beginPath();
          ctx.arc(pos.x, pos.y, 9, 0, Math.PI * 2);
          ctx.fillStyle = 'rgba(74, 222, 128, 0.25)';
          ctx.fill();

          ctx.beginPath();
          ctx.arc(pos.x, pos.y, 6, 0, Math.PI * 2);
          ctx.strokeStyle = '#86efac';
          ctx.lineWidth = 2;
          ctx.stroke();
        }

        ctx.beginPath();
        ctx.arc(pos.x, pos.y, isTopMatch ? 5 : 4, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
      }

      // Query node
      if (projectedQuery2D) {
        const qPos = toCanvasCoords(projectedQuery2D.x, projectedQuery2D.y);
        ctx.beginPath();
        ctx.arc(qPos.x, qPos.y, 14, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(239, 68, 68, 0.2)';
        ctx.fill();

        ctx.beginPath();
        ctx.arc(qPos.x, qPos.y, 6, 0, Math.PI * 2);
        ctx.fillStyle = '#ef4444';
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.stroke();

        ctx.fillStyle = '#fca5a5';
        ctx.font = 'bold 11px JetBrains Mono, monospace';
        ctx.textAlign = 'left';
        ctx.fillText('Q (Query)', qPos.x + 10, qPos.y - 8);
      }
    } else {
      // 3D Perspective Orbit Projection
      const fov = 300 * zoom;
      const cameraZ = 2.5;

      const project3DToScreen = (x3d: number, y3d: number, z3d: number) => {
        // Rotate around Y (yaw)
        const cosY = Math.cos(rotation.yaw);
        const sinY = Math.sin(rotation.yaw);
        const x1 = x3d * cosY - z3d * sinY;
        const z1 = x3d * sinY + z3d * cosY;

        // Rotate around X (pitch)
        const cosX = Math.cos(rotation.pitch);
        const sinX = Math.sin(rotation.pitch);
        const y2 = y3d * cosX - z1 * sinX;
        const z2 = y3d * sinX + z1 * cosX;

        const distance = z2 + cameraZ;
        const scale = fov / Math.max(0.1, distance);

        return {
          screenX: width / 2 + pan.x + x1 * scale,
          screenY: height / 2 + pan.y + y2 * scale,
          scale,
          depth: distance,
        };
      };

      // 3D coordinate box outline
      ctx.strokeStyle = 'rgba(51, 65, 85, 0.3)';
      ctx.lineWidth = 1;
      const corners = [
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
      ].map(([cx, cy, cz]) => project3DToScreen(cx, cy, cz));

      const edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
      ];

      for (const [from, to] of edges) {
        ctx.beginPath();
        ctx.moveTo(corners[from].screenX, corners[from].screenY);
        ctx.lineTo(corners[to].screenX, corners[to].screenY);
        ctx.stroke();
      }

      // Calculate screen positions and sort by depth (back to front)
      const rendered3D = projectedPoints3D.map((pt) => {
        const proj = project3DToScreen(pt.x, pt.y, pt.z || 0);
        pt.screenX = proj.screenX;
        pt.screenY = proj.screenY;
        pt.depth = proj.depth;
        return { pt, proj };
      });

      rendered3D.sort((a, b) => b.proj.depth - a.proj.depth);

      // Query line connections in 3D
      if (projectedQuery3D) {
        const qProj = project3DToScreen(projectedQuery3D.x, projectedQuery3D.y, projectedQuery3D.z);
        for (const item of rendered3D) {
          if (topResultIds.has(item.pt.id)) {
            ctx.beginPath();
            ctx.moveTo(qProj.screenX, qProj.screenY);
            ctx.lineTo(item.proj.screenX, item.proj.screenY);
            ctx.strokeStyle = 'rgba(74, 222, 128, 0.4)';
            ctx.lineWidth = 1.5;
            ctx.setLineDash([3, 3]);
            ctx.stroke();
            ctx.setLineDash([]);
          }
        }
      }

      // Render 3D points
      for (const { pt, proj } of rendered3D) {
        const isTopMatch = topResultIds.has(pt.id);
        const color = docColorMap.get(pt.docName) || '#22c55e';
        const baseRadius = Math.max(2, Math.min(10, 5 * (proj.scale / 150)));

        if (isTopMatch) {
          ctx.beginPath();
          ctx.arc(proj.screenX, proj.screenY, baseRadius + 4, 0, Math.PI * 2);
          ctx.fillStyle = 'rgba(74, 222, 128, 0.3)';
          ctx.fill();
        }

        ctx.beginPath();
        ctx.arc(proj.screenX, proj.screenY, baseRadius, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
      }

      // Query node in 3D
      if (projectedQuery3D) {
        const qProj = project3DToScreen(projectedQuery3D.x, projectedQuery3D.y, projectedQuery3D.z);
        ctx.beginPath();
        ctx.arc(qProj.screenX, qProj.screenY, 7, 0, Math.PI * 2);
        ctx.fillStyle = '#ef4444';
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }
  }, [
    viewMode,
    projectedPoints2D,
    projectedPoints3D,
    projectedQuery2D,
    projectedQuery3D,
    topResultIds,
    docColorMap,
    zoom,
    pan,
    rotation,
  ]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (isDragging) {
      if (viewMode === '3D') {
        const dx = e.clientX - dragStart.x;
        const dy = e.clientY - dragStart.y;
        setRotation((prev) => ({
          yaw: prev.yaw + dx * 0.008,
          pitch: Math.max(-1.4, Math.min(1.4, prev.pitch + dy * 0.008)),
        }));
        setDragStart({ x: e.clientX, y: e.clientY });
      } else {
        setPan({
          x: e.clientX - dragStart.x,
          y: e.clientY - dragStart.y,
        });
      }
      return;
    }

    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const points = viewMode === '2D' ? projectedPoints2D : projectedPoints3D;
    let closest: ProjectedPoint | null = null;
    let minDist = 18;

    for (const pt of points) {
      if (pt.screenX !== undefined && pt.screenY !== undefined) {
        const d = Math.hypot(mouseX - pt.screenX, mouseY - pt.screenY);
        if (d < minDist) {
          minDist = d;
          closest = pt;
        }
      }
    }

    if (closest) {
      setHoveredPoint({
        point: closest,
        screenX: mouseX,
        screenY: mouseY,
      });
    } else {
      setHoveredPoint(null);
    }
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    setDragStart({
      x: e.clientX - (viewMode === '2D' ? pan.x : 0),
      y: e.clientY - (viewMode === '2D' ? pan.y : 0),
    });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const resetView = () => {
    setZoom(1.0);
    setPan({ x: 0, y: 0 });
    setRotation({ yaw: 0.4, pitch: 0.2 });
  };

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 backdrop-blur-sm relative overflow-hidden">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2">
            <Eye className="w-5 h-5 text-brand-400" />
            <h2 className="text-base font-semibold text-white">Vector Space Projection</h2>
          </div>

          {/* 2D vs 3D View Toggle */}
          <div className="flex items-center space-x-1 bg-slate-950 p-1 rounded-xl border border-slate-800 text-xs font-mono">
            <button
              type="button"
              onClick={() => setViewMode('2D')}
              className={`flex items-center space-x-1 px-2.5 py-1 rounded-lg transition-all ${
                viewMode === '2D'
                  ? 'bg-brand-500 text-slate-950 font-bold'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <Box className="w-3.5 h-3.5" />
              <span>2D Map</span>
            </button>
            <button
              type="button"
              onClick={() => setViewMode('3D')}
              className={`flex items-center space-x-1 px-2.5 py-1 rounded-lg transition-all ${
                viewMode === '3D'
                  ? 'bg-brand-500 text-slate-950 font-bold'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <Orbit className="w-3.5 h-3.5" />
              <span>3D Orbit</span>
            </button>
          </div>
        </div>

        <div className="flex items-center space-x-1.5 bg-slate-950 px-2 py-1 rounded-xl border border-slate-800 text-xs">
          <button
            type="button"
            onClick={() => setZoom((z) => Math.min(3.0, z * 1.2))}
            title="Zoom In"
            className="p-1 text-slate-400 hover:text-white rounded hover:bg-slate-800 transition-colors"
          >
            <ZoomIn className="w-4 h-4" />
          </button>
          <button
            type="button"
            onClick={() => setZoom((z) => Math.max(0.4, z / 1.2))}
            title="Zoom Out"
            className="p-1 text-slate-400 hover:text-white rounded hover:bg-slate-800 transition-colors"
          >
            <ZoomOut className="w-4 h-4" />
          </button>
          <button
            type="button"
            onClick={resetView}
            title="Reset View"
            className="p-1 text-slate-400 hover:text-white rounded hover:bg-slate-800 transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      <div className="relative border border-slate-800 rounded-xl overflow-hidden cursor-crosshair">
        <canvas
          ref={canvasRef}
          width={800}
          height={400}
          onMouseMove={handleMouseMove}
          onMouseDown={handleMouseDown}
          onMouseUp={handleMouseUp}
          onClick={() => hoveredPoint && onSelectChunk && onSelectChunk(hoveredPoint.point.chunk)}
          className="w-full h-[360px] block"
        />

        {hoveredPoint && (
          <div
            className="absolute z-30 pointer-events-none p-3 rounded-xl bg-slate-950/95 border border-slate-700 shadow-2xl backdrop-blur-md max-w-xs text-xs font-mono"
            style={{
              left: Math.min(hoveredPoint.screenX + 15, 540),
              top: Math.max(10, hoveredPoint.screenY - 10),
            }}
          >
            <div className="flex items-center justify-between mb-1 text-[11px] text-brand-400 font-bold">
              <span>{hoveredPoint.point.docName}</span>
              <span className="text-slate-400">Chunk {hoveredPoint.point.chunk.chunkIndex}</span>
            </div>
            <p className="text-slate-300 line-clamp-3 font-sans leading-relaxed">
              {hoveredPoint.point.chunk.text}
            </p>
            <div className="mt-1.5 pt-1.5 border-t border-slate-800 text-[10px] text-slate-500 flex justify-between">
              <span>Click point to inspect</span>
              <span>~{hoveredPoint.point.chunk.tokenCountEstimate} tokens</span>
            </div>
          </div>
        )}

        <div className="absolute bottom-3 left-3 bg-slate-950/80 border border-slate-800/80 rounded-lg px-3 py-1.5 backdrop-blur-md flex items-center space-x-3 text-[11px]">
          {Array.from(docColorMap.entries()).map(([name, color]) => (
            <div key={name} className="flex items-center space-x-1.5">
              <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }} />
              <span className="text-slate-300 font-mono truncate max-w-[120px]">{name}</span>
            </div>
          ))}
          <span className="text-[10px] text-slate-500 font-mono pl-2 border-l border-slate-800">
            {viewMode === '3D' ? 'Drag to rotate 3D orbit' : 'Pan & hover to explore'}
          </span>
        </div>
      </div>
    </div>
  );
};
