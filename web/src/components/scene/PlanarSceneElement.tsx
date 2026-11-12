import type { SVGProps } from 'react';
import type { ResolvedSceneElement } from '@/features/scenario/projections';
import { semanticTokens } from '@/components/ui/semanticTokens';
import type { ScenePoint } from './planarProjection';

export function PlanarSceneElement({
  element,
  active,
  project,
  scale,
  interaction,
}: {
  element: ResolvedSceneElement;
  active: boolean;
  project: (point: ScenePoint) => ScenePoint;
  scale: number;
  interaction: SVGProps<SVGGElement>;
}) {
  const stroke = active ? '#0f766e' : '#64748b';
  const fill = active ? '#ccfbf1' : '#f8fafc';
  const markerRadius = element.scale_invariant ? 7 : Math.max(4, Math.min(18, scale * 0.02));

  if (element.geometry.kind === 'bounds') {
    const [x0, y0] = project(element.geometry.min);
    const [x1, y1] = project(element.geometry.max);
    return (
      <g {...interaction}>
        <rect
          x={Math.min(x0, x1)}
          y={Math.min(y0, y1)}
          width={Math.abs(x1 - x0)}
          height={Math.abs(y1 - y0)}
          fill={active ? '#dcfce7' : '#f0fdf4'}
          stroke={active ? '#16a34a' : '#86efac'}
          strokeWidth={1.5}
          vectorEffect="non-scaling-stroke"
        />
      </g>
    );
  }

  if (element.geometry.kind === 'polyline') {
    const geometry = element.geometry;
    const points = geometry.points.map(project).map((point) => point.join(',')).join(' ');
    return (
      <g {...interaction}>
        <polyline
          points={points}
          fill="none"
          stroke={active ? '#b45309' : stroke}
          strokeWidth={active ? 8 : 6}
          strokeLinecap="round"
          strokeLinejoin="round"
          vectorEffect="non-scaling-stroke"
        />
        {geometry.points.map((point, index) => {
          const [x, y] = project(point);
          return (
            <circle
              key={`${element.id}:joint:${index}`}
              cx={x}
              cy={y}
              r={index === geometry.points.length - 1 ? markerRadius : 4.5}
              fill={index === geometry.points.length - 1 ? '#475569' : '#ffffff'}
              stroke={active ? '#92400e' : '#64748b'}
              strokeWidth={1.5}
              vectorEffect="non-scaling-stroke"
            />
          );
        })}
      </g>
    );
  }

  if (element.geometry.kind === 'points') {
    const glyph = String(element.style.glyph ?? element.metadata.glyph ?? '');
    return (
      <g {...interaction}>
        {element.geometry.points.map((point, index) => {
          const [x, y] = project(point);
          const target = glyph === 'target' || element.metadata.canonical_goal === true;
          return (
            <g key={`${element.id}:point:${index}`}>
              {target ? (
                <>
                  <circle
                    cx={x}
                    cy={y}
                    r={markerRadius + 3}
                    fill="none"
                    stroke={active ? '#059669' : '#34d399'}
                    strokeWidth={1.5}
                    vectorEffect="non-scaling-stroke"
                  />
                  <circle
                    cx={x}
                    cy={y}
                    r={markerRadius * 0.45}
                    fill={active ? '#059669' : '#a7f3d0'}
                  />
                </>
              ) : (
                <circle
                  cx={x}
                  cy={y}
                  r={markerRadius}
                  fill={fill}
                  stroke={stroke}
                  strokeWidth={1.5}
                  vectorEffect="non-scaling-stroke"
                />
              )}
            </g>
          );
        })}
      </g>
    );
  }

  if (element.geometry.kind === 'link') {
    const points = element.geometry.points.map(project);
    if (points.length < 2) return null;
    return (
      <g {...interaction}>
        <line
          x1={points[0][0]}
          y1={points[0][1]}
          x2={points[1][0]}
          y2={points[1][1]}
          stroke={active ? semanticTokens.objective.strongStroke : semanticTokens.objective.stroke}
          strokeWidth={active ? 3 : 2.25}
          strokeDasharray="6 5"
          opacity={active ? 0.95 : 0.72}
          vectorEffect="non-scaling-stroke"
        />
      </g>
    );
  }

  return null;
}
