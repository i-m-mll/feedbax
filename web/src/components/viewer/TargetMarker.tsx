import { useMemo } from 'react';
import { Line } from '@react-three/drei';

const TASK_COLORS: Record<number, string> = {
  0: '#2fbf7f', // Reach — green/mint
  1: '#f2b92d', // Hold — amber
  2: '#9b59b6', // Track — purple
  3: '#2f7cf6', // Swing — blue/brand
};

const CROSS_SIZE = 0.015;

interface TargetMarkerProps {
  position: [number, number];
  taskType: number;
  trace?: number[][];
}

export function TargetMarker({ position, taskType, trace }: TargetMarkerProps) {
  const color = TASK_COLORS[taskType] ?? '#9b9cab';
  const [x, y] = position;

  // Cross lines for the target marker
  const hLine = useMemo(
    (): [number, number, number][] => [
      [x - CROSS_SIZE, y, 0],
      [x + CROSS_SIZE, y, 0],
    ],
    [x, y],
  );

  const vLine = useMemo(
    (): [number, number, number][] => [
      [x, y - CROSS_SIZE, 0],
      [x, y + CROSS_SIZE, 0],
    ],
    [x, y],
  );

  // Target trace as 3D points (z=0)
  const tracePoints = useMemo(() => {
    if (!trace || trace.length < 2) return null;
    return trace.map(
      (p): [number, number, number] => [p[0], p[1], 0],
    );
  }, [trace]);

  return (
    <group>
      <Line points={hLine} color={color} lineWidth={2.5} />
      <Line points={vLine} color={color} lineWidth={2.5} />
      <mesh position={[x, y, 0]}>
        <sphereGeometry args={[0.005, 12, 12]} />
        <meshStandardMaterial color={color} transparent opacity={0.6} />
      </mesh>
      {tracePoints && (
        <Line
          points={tracePoints}
          color={color}
          lineWidth={1.5}
          transparent
          opacity={0.35}
        />
      )}
    </group>
  );
}