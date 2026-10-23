import { useCallback, useEffect, useRef, useState } from "react";
import {
  launchInstance,
  authenticateInstanceReservation,
  fetchOrchestrationStatus,
  terminateInstance,
  type LaunchInstanceRequest,
} from "@/api/client";
import { useTrainingStore } from "@/stores/trainingStore";

const POLL_INTERVAL_MS = 5000;

/** Terminal statuses that stop polling. */
const TERMINAL_STATUSES = new Set(["idle", "running", "preempted", "error"]);

export function useOrchestration() {
  const setOrchestrationState = useTrainingStore(
    (s) => s.setOrchestrationState,
  );
  const setWorkerConfig = useTrainingStore((s) => s.setWorkerConfig);
  const orchestrationStatus = useTrainingStore((s) => s.orchestrationStatus);
  const orchestrationInstanceName = useTrainingStore(
    (s) => s.orchestrationInstanceName,
  );
  const orchestrationWorkerUrl = useTrainingStore(
    (s) => s.orchestrationWorkerUrl,
  );

  const [launching, setLaunching] = useState(false);
  const [terminating, setTerminating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [intentId, setIntentId] = useState<string | null>(null);
  const [reservation, setReservation] = useState<{
    id: string;
    expiresAt: string;
    maximumCost: number;
  } | null>(null);

  const pollTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pollingRef = useRef(false);

  const stopPolling = useCallback(() => {
    if (pollTimerRef.current !== null) {
      clearTimeout(pollTimerRef.current);
      pollTimerRef.current = null;
    }
    pollingRef.current = false;
  }, []);

  const pollStatus = useCallback(async () => {
    if (!pollingRef.current) return;
    try {
      const res = await fetchOrchestrationStatus(intentId);
      setOrchestrationState(res.status, res.instance_name, res.worker_url);
      if (res.error) {
        setError(res.error);
      }
      // When the worker is running, also sync the TrainingService worker config.
      if (res.status === "running" && res.worker_url) {
        setWorkerConfig("remote", res.worker_url, true);
      }
      // Stop polling once we reach a terminal state.
      if (TERMINAL_STATUSES.has(res.status)) {
        stopPolling();
        return;
      }
    } catch {
      // Ignore transient fetch errors during polling.
    }
    if (pollingRef.current) {
      pollTimerRef.current = setTimeout(pollStatus, POLL_INTERVAL_MS);
    }
  }, [intentId, setOrchestrationState, setWorkerConfig, stopPolling]);

  const startPolling = useCallback(() => {
    stopPolling();
    pollingRef.current = true;
    pollTimerRef.current = setTimeout(pollStatus, POLL_INTERVAL_MS);
  }, [pollStatus, stopPolling]);

  // Clean up on unmount.
  useEffect(() => {
    return () => stopPolling();
  }, [stopPolling]);

  const launch = useCallback(
    async (params: LaunchInstanceRequest) => {
      setLaunching(true);
      setError(null);
      try {
        const res = await launchInstance(params);
        setIntentId(res.intent_id);
        setReservation({
          id: res.reservation_id,
          expiresAt: res.expires_at,
          maximumCost: res.expected_cost.maximum,
        });
        setOrchestrationState(res.status, res.instance_name, null);
      } catch (err) {
        const msg =
          err instanceof Error ? err.message : "Failed to launch instance";
        setError(msg);
        setOrchestrationState("error", null, null);
      } finally {
        setLaunching(false);
      }
    },
    [setOrchestrationState, startPolling],
  );

  const authenticate = useCallback(
    async (operatorIdentity: string) => {
      if (!intentId || !reservation) {
        throw new Error("No effect reservation is awaiting authentication");
      }
      setLaunching(true);
      setError(null);
      try {
        await authenticateInstanceReservation(intentId, reservation.id, {
          operator_identity: operatorIdentity,
          authentication_id: crypto.randomUUID(),
          confirmation_token: "launch-billable-gcp-worker",
          max_cost_usd: reservation.maximumCost,
        });
        setOrchestrationState("authenticated", orchestrationInstanceName, null);
        startPolling();
      } catch (err) {
        const msg =
          err instanceof Error
            ? err.message
            : "Failed to authenticate reservation";
        setError(msg);
      } finally {
        setLaunching(false);
      }
    },
    [
      intentId,
      orchestrationInstanceName,
      reservation,
      setOrchestrationState,
      startPolling,
    ],
  );

  const terminate = useCallback(async () => {
    setTerminating(true);
    setError(null);
    stopPolling();
    try {
      await terminateInstance(intentId);
      setOrchestrationState("idle", null, null);
      // Disconnect the worker config.
      setWorkerConfig("local", null, false);
    } catch (err) {
      const msg =
        err instanceof Error ? err.message : "Failed to terminate instance";
      setError(msg);
    } finally {
      setTerminating(false);
    }
  }, [intentId, setOrchestrationState, setWorkerConfig, stopPolling]);

  return {
    status: orchestrationStatus,
    instanceName: orchestrationInstanceName,
    workerUrl: orchestrationWorkerUrl,
    intentId,
    reservation,
    launching,
    terminating,
    error,
    launch,
    authenticate,
    terminate,
  };
}
