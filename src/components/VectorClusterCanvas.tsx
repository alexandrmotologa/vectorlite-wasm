import React, { useRef, useEffect, useState, useMemo } from 'react';
import { FastPCA, Point2D } from '../engine/pca';
import { TextChunk } from '../engine/chunker';
import { HybridSearchResult } from '../engine/hybrid';
import { Eye, RefreshCw, ZoomIn, ZoomOut } from 'lucide-react';

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
  docName: string;
  chunk: TextChunk;
}

// Color palette for documents
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
  const [hoveredPoint, setHoveredPoint] = useState<{ point: ProjectedPoint; screenX: number; screenY: number } | null>(null);
  const [zoom, setZoom] = useState(1.0);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  // Map document names to colors
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
  const { pcaModel, projectedPoints } = useMemo(() => {
    if (vectors.length < 2) {
      return { pcaModel: null, projectedPoints: [] };
    }

    try {
      const rawVecs = vectors.map((v) => v.vector);
      const model = FastPCA.fit(rawVecs, 15);

      const points: ProjectedPoint[] = vectors.map((v) => {
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

      return { pcaModel: model, projectedPoints: points };
    } catch {
      return { pcaModel: null, projectedPoints: [] };
    }
  }, [vectors, chunksMap]);

  // Project query vector if present
  const projectedQuery: Point2D | null = useMemo(() => {
    if (!queryVector || !pcaModel) return null;
    return FastPCA.project(queryVector, pcaModel);
  }, [queryVector, pcaModel]);

  // Top result IDs for radar highlighting
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

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Draw dark background grid
    ctx.fillStyle = '#050914';
    ctx.fillRect(0, 0, width, height);

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

    if (projectedPoints.length === 0) {
      ctx.fillStyle = '#64748b';
      ctx.font = '13px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Load documents above to project embeddings in 2D vector space', width / 2, height / 2);
      return;
    }

    // Coordinate conversion function
    const toCanvasCoords = (ptX: number, ptY: number) => {
      const cx = width / 2 + pan.x;
      const cy = height / 2 + pan.y;
      const x = cx + (ptX - 0.5) * plotWidth * zoom;
      const y = cy + (ptY - 0.5) * plotHeight * zoom;
      return { x, y };
    };

    // 1. Draw connecting lines from query to top-K results
    if (projectedQuery) {
      const qPos = toCanvasCoords(projectedQuery.x, projectedQuery.y);

      // Radiating radar pulse rings
      ctx.strokeStyle = 'rgba(34, 197, 94, 0.15)';
      ctx.lineWidth = 1.5;
      [40, 80, 130, 190].forEach((r) => {
        ctx.beginPath();
        ctx.arc(qPos.x, qPos.y, r * zoom, 0, Math.PI * 2);
        ctx.stroke();
      });

      for (const pt of projectedPoints) {
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

    // 2. Draw Vector Points
    for (const pt of projectedPoints) {
      const pos = toCanvasCoords(pt.x, pt.y);
      const isTopMatch = topResultIds.has(pt.id);
      const color = docColorMap.get(pt.docName) || '#22c55e';

      // Outer glow for top matches
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

      // Point circle
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, isTopMatch ? 5 : 4, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
    }

    // 3. Draw Query Point if active
    if (projectedQuery) {
      const qPos = toCanvasCoords(projectedQuery.x, projectedQuery.y);

      // Glow halo
      ctx.beginPath();
      ctx.arc(qPos.x, qPos.y, 14, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(239, 68, 68, 0.2)';
      ctx.fill();

      // Query center node
      ctx.beginPath();
      ctx.arc(qPos.x, qPos.y, 6, 0, Math.PI * 2);
      ctx.fillStyle = '#ef4444';
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Label
      ctx.fillStyle = '#fca5a5';
      ctx.font = 'bold 11px JetBrains Mono, monospace';
      ctx.textAlign = 'left';
      ctx.fillText('Q (Query)', qPos.x + 10, qPos.y - 8);
    }
  }, [projectedPoints, projectedQuery, topResultIds, docColorMap, zoom, pan]);

  // Mouse hover detection
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (isDragging) {
      setPan({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y,
      });
      return;
    }

    const canvas = canvasRef.current;
    if (!canvas || projectedPoints.length === 0) return;
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const width = canvas.width;
    const height = canvas.height;
    const padding = 40;
    const plotWidth = width - padding * 2;
    const plotHeight = height - padding * 2;

    let closest: ProjectedPoint | null = null;
    let minDist = 16; // Hit test radius in pixels

    for (const pt of projectedPoints) {
      const cx = width / 2 + pan.x;
      const cy = height / 2 + pan.y;
      const ptX = cx + (pt.x - 0.5) * plotWidth * zoom;
      const ptY = cy + (pt.y - 0.5) * plotHeight * zoom;

      const d = Math.hypot(mouseX - ptX, mouseY - ptY);
      if (d < minDist) {
        minDist = d;
        closest = pt;
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
      x: e.clientX - pan.x,
      y: e.clientY - pan.y,
    });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleCanvasClick = () => {
    if (hoveredPoint && onSelectChunk) {
      onSelectChunk(hoveredPoint.point.chunk);
    }
  };

  const resetView = () => {
    setZoom(1.0);
    setPan({ x: 0, y: 0 });
  };

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 backdrop-blur-sm relative overflow-hidden">
      {/* Header & Controls */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2">
          <Eye className="w-5 h-5 text-brand-400" />
          <h2 className="text-base font-semibold text-white">2D Vector Space Projection (PCA)</h2>
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

      {/* Canvas Area */}
      <div className="relative border border-slate-800 rounded-xl overflow-hidden cursor-crosshair">
        <canvas
          ref={canvasRef}
          width={800}
          height={400}
          onMouseMove={handleMouseMove}
          onMouseDown={handleMouseDown}
          onMouseUp={handleMouseUp}
          onClick={handleCanvasClick}
          className="w-full h-[360px] block"
        />

        {/* Hover Tooltip */}
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

        {/* Legend */}
        <div className="absolute bottom-3 left-3 bg-slate-950/80 border border-slate-800/80 rounded-lg px-3 py-1.5 backdrop-blur-md flex items-center space-x-3 text-[11px]">
          {Array.from(docColorMap.entries()).map(([name, color]) => (
            <div key={name} className="flex items-center space-x-1.5">
              <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }} />
              <span className="text-slate-300 font-mono truncate max-w-[120px]">{name}</span>
            </div>
          ))}
          {projectedQuery && (
            <div className="flex items-center space-x-1.5 border-l border-slate-800 pl-2">
              <span className="w-2.5 h-2.5 rounded-full bg-red-500" />
              <span className="text-red-300 font-mono">Query Radar</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
