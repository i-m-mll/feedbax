/**
 * Compute joint positions from relative joint angles and segment lengths.
 * Angles are relative (local frame), in radians. Planar (z=0).
 * Returns n_joints+1 positions (including root at origin).
 */
export function forwardKinematics(
  jointAngles: number[],
  segmentLengths: number[],
): [number, number, number][] {
  const positions: [number, number, number][] = [[0, 0, 0]];
  let cumulativeAngle = 0;
  const n = Math.min(jointAngles.length, segmentLengths.length);
  for (let i = 0; i < n; i++) {
    cumulativeAngle += jointAngles[i];
    const prev = positions[positions.length - 1];
    positions.push([
      prev[0] + segmentLengths[i] * Math.cos(cumulativeAngle),
      prev[1] + segmentLengths[i] * Math.sin(cumulativeAngle),
      0,
    ]);
  }
  return positions;
}
