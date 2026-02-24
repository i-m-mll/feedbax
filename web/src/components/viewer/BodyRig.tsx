import { useMemo } from 'react';
import { Line } from '@react-three/drei';
import { forwardKinematics } from '@/utils/kinematics';

interface BodyRigProps {
  jointAngles: number[];
  segmentLengths: number[];
  effectorPos: number[];
  effectorTrace: number[][];
}

export function BodyRig({ jointAngles, segmentLengths, effectorTrace }: BodyRigProps) {
  const positions = useMemo(
    () => forwardKinematics(jointAngles, segmentLengths),
    [jointAngles, segmentLengths],
  );

  // Build segment pairs for lines
  const segmentPoints = useMemo(() => {
    const pairs: [number, number, number][][] = [];
    for (let i = 0; i < positions.length - 1; i++) {
      pairs.push([positions[i], positions[i + 1]]);
    }
    return pairs;
  }, [positions]);

  // Effector trace as 3D points (z=0)
  const tracePoints = useMemo(() => {
    if (effectorTrace.length < 2) return null;
    return effectorTrace.map(
      (p): [number, number, number] => [p[0], p[1], 0],
    );
  }, [effectorTrace]);

  const effectorPosition = positions[positions.length - 1];

  return (
    <group>
      {/* Arm segments */}
      {segmentPoints.map((pair, i) => (
        <Line
          key={`seg-${i}`}
          points={pair}
          color="#9b9cab"
          lineWidth={4}
        />
      ))}

      {/* Joint spheres */}
      {positions.slice(0, -1).map((pos, i) => (
        <mesh key={`joint-${i}`} position={pos}>
          <sphereGeometry args={[0.008, 16, 16]} />
          <meshStandardMaterial color="#44465a" />
        </mesh>
      ))}

      {/* Effector sphere (larger, brand color) */}
      <mesh position={effectorPosition}>
        <sphereGeometry args={[0.012, 16, 16]} />
        <meshStandardMaterial color="#2f7cf6" />
      </mesh>

      {/* Effector trace */}
      {tracePoints && (
        <Line
          points={tracePoints}
          color="#2f7cf6"
          lineWidth={1.5}
          transparent
          opacity={0.4}
        />
      )}
    </group>
  );
}
