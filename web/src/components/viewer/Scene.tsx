import { useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import type { TrajectoryData } from '@/types/trajectory';
import { BodyRig } from './BodyRig';
import { TargetMarker } from './TargetMarker';

// Default segment lengths for a 3-link planar arm (can be overridden via props)
const DEFAULT_SEGMENT_LENGTHS = [0.3, 0.33, 0.16];
const TRACE_WINDOW = 100;

interface SceneProps {
  trajectoryData: TrajectoryData | null;
  frame: number;
  segmentLengths?: number[];
}

/**
 * XY-plane grid helper.
 * Three.js GridHelper is on XZ by default, so we rotate 90 degrees around X.
 */
function XYGrid() {
  return (
    <gridHelper
      args={[1.2, 24, '#d9d9df', '#ededf0']}
      rotation={[Math.PI / 2, 0, 0]}
      position={[0.3, 0.3, -0.01]}
    />
  );
}

export function Scene({ trajectoryData, frame, segmentLengths }: SceneProps) {
  const lengths = segmentLengths ?? DEFAULT_SEGMENT_LENGTHS;
  const frameIdx = Math.max(0, Math.min(Math.floor(frame), (trajectoryData?.timestamps.length ?? 1) - 1));

  const jointAngles = trajectoryData?.joint_angles[frameIdx] ?? [];
  const effectorPos = trajectoryData?.effector_pos[frameIdx] ?? [0, 0];
  const taskTarget = trajectoryData?.task_target[frameIdx] ?? [0, 0];
  const taskType = trajectoryData?.task_type ?? 0;

  // Build effector trace from past frames
  const effectorTrace = useMemo(() => {
    if (!trajectoryData) return [];
    const start = Math.max(0, frameIdx - TRACE_WINDOW);
    return trajectoryData.effector_pos.slice(start, frameIdx + 1);
  }, [trajectoryData, frameIdx]);

  return (
    <Canvas
      camera={{
        position: [0.3, 0.3, 1.5],
        near: 0.01,
        far: 100,
        fov: 35,
      }}
      style={{ background: 'transparent' }}
      gl={{ alpha: true }}
    >
      <ambientLight intensity={0.6} />
      <directionalLight position={[5, 5, 5]} intensity={0.8} />

      <OrbitControls
        target={[0.3, 0.3, 0]}
        enablePan
        enableZoom
        enableRotate
      />

      <XYGrid />

      {trajectoryData && (
        <>
          <BodyRig
            jointAngles={jointAngles}
            segmentLengths={lengths}
            effectorPos={effectorPos}
            effectorTrace={effectorTrace}
          />
          <TargetMarker
            position={[taskTarget[0], taskTarget[1]]}
            taskType={taskType}
          />
        </>
      )}
    </Canvas>
  );
}
