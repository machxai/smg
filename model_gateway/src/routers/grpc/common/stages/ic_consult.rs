//! Inference Cache (IC) route consult.
//!
//! IC is the routing brain. When enabled (`RouterConfig::ic_lookup`), the gRPC
//! regular pipeline makes ONE time-boxed `LookupRoute` gRPC call with the
//! request's own `token_ids` and obeys the returned best replica. There is no
//! blending in SMG: on any miss / timeout / error the existing routing policy
//! (`consistent_hashing`) decides. The consult is **fail-open** — it never
//! blocks or fails the request path beyond its bounded deadline.
//!
//! This module owns two concerns:
//!   - [`IcConsultant`]: config + a lazily-connected [`ic_client::IcClient`],
//!     exposing a single fail-open [`IcConsultant::consult`].
//!   - [`resolve_replica_to_available_index`]: the replica-identity seam that
//!     maps an IC `replica_id` to a positional index into the current worker
//!     slice — the same positional contract the internal `x-smg-target-worker`
//!     mechanism uses. See the doc comment there for the open identity problem.

use std::{sync::Arc, time::Duration};

use ic_client::{IcClient, IcClientError, RankedReplica};
use tracing::debug;

use crate::{config::IcLookupConfig, worker::Worker};

/// Wraps a lazily-connected IC gRPC client plus the per-deployment lookup
/// parameters (tenant, hash scheme, deadline).
pub(crate) struct IcConsultant {
    client: IcClient,
    tenant_id: String,
    hash_scheme: String,
    timeout: Duration,
}

impl std::fmt::Debug for IcConsultant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IcConsultant")
            .field("endpoint", &self.client.endpoint())
            .field("tenant_id", &self.tenant_id)
            .field("hash_scheme", &self.hash_scheme)
            .field("timeout", &self.timeout)
            .finish()
    }
}

impl IcConsultant {
    /// Build a consultant from config. Uses a lazily-connecting channel: an
    /// unreachable IC must never block or fail router startup — a failed dial
    /// surfaces as a per-request (fail-open) error instead.
    pub(crate) fn from_config(config: &IcLookupConfig) -> Result<Self, IcClientError> {
        let timeout = Duration::from_millis(config.timeout_ms);
        // A dial that can't complete within the per-lookup budget is useless, so
        // bound the connect timeout by the same deadline.
        let client = IcClient::connect_lazy(config.endpoint.clone(), timeout)?;
        Ok(Self {
            client,
            tenant_id: config.tenant_id.clone(),
            hash_scheme: config.hash_scheme.clone(),
            timeout,
        })
    }

    /// Consult IC for a ranked replica list, bounded by the configured deadline.
    ///
    /// Fail-open: returns `None` on an empty input, a `NO_HINT` / empty response,
    /// a timeout, or any transport error — the caller then routes with its normal
    /// policy. A returned `Some` is best-first (IC ranks by score).
    pub(crate) async fn consult(
        &self,
        model_id: &str,
        token_ids: &[u32],
    ) -> Option<Vec<RankedReplica>> {
        if token_ids.is_empty() {
            return None;
        }

        // token_ids must be owned for the proto request; this clone only happens
        // when IC is enabled and the active policy honors the hint.
        let result = self
            .client
            .lookup_route(
                model_id.to_string(),
                self.tenant_id.clone(),
                self.hash_scheme.clone(),
                token_ids.to_vec(),
                self.timeout,
            )
            .await;

        match result {
            Ok(hint) if hint.is_no_hint() => {
                debug!(
                    model_id,
                    reason = %hint.reason_code,
                    "IC lookup returned no usable hint; using policy"
                );
                None
            }
            Ok(hint) => {
                debug!(
                    model_id,
                    reason = %hint.reason_code,
                    replicas = hint.replicas.len(),
                    latency_us = hint.lookup_latency_us,
                    "IC lookup hint received"
                );
                Some(hint.replicas)
            }
            Err(err) => {
                debug!(
                    model_id,
                    error = %err,
                    "IC lookup failed; failing open to policy"
                );
                None
            }
        }
    }
}

/// Resolve the best usable IC replica hint to a positional index into the
/// `available` worker slice — the same positional contract the internal
/// `x-smg-target-worker` mechanism uses.
///
/// ## Replica-identity seam — KNOWN OPEN PROBLEM (stable identity is a separate ticket)
///
/// IC returns an engine-defined `replica_id`; SMG workers are keyed by URL. No
/// stable identity contract exists between the two yet. Until it does, this
/// matches `replica_id` against, in order:
///   1. the worker URL (exact), then
///   2. a `replica_id` worker label (exact),
/// walking the ranked list best-first and returning the first replica that maps
/// to a worker in the `available` slice. `None` (no match) means "omit the
/// target" so the caller falls back to its policy. This function is the single
/// place to swap in the real identity mapping when the contract lands.
pub(crate) fn resolve_replica_to_available_index(
    ranked: &[RankedReplica],
    available: &[Arc<dyn Worker>],
) -> Option<usize> {
    ranked.iter().find_map(|replica| {
        available
            .iter()
            .position(|w| worker_matches_replica(w, &replica.replica_id))
    })
}

/// Whether `worker` is the target named by `replica_id` under the current
/// (best-effort) identity seam: exact URL, else a matching `replica_id` label.
fn worker_matches_replica(worker: &Arc<dyn Worker>, replica_id: &str) -> bool {
    if worker.url() == replica_id {
        return true;
    }
    worker
        .metadata()
        .spec
        .labels
        .get("replica_id")
        .is_some_and(|label| label == replica_id)
}

#[cfg(test)]
mod tests {
    use openai_protocol::worker::HealthCheckConfig;

    use super::*;
    use crate::worker::{BasicWorkerBuilder, WorkerType};

    fn no_health_check() -> HealthCheckConfig {
        HealthCheckConfig {
            disable_health_check: true,
            ..Default::default()
        }
    }

    fn worker(url: &str) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .health_config(no_health_check())
                .build(),
        )
    }

    fn worker_with_replica_label(url: &str, replica_id: &str) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .label("replica_id", replica_id)
                .health_config(no_health_check())
                .build(),
        )
    }

    fn ranked(ids: &[&str]) -> Vec<RankedReplica> {
        ids.iter()
            .enumerate()
            .map(|(i, id)| RankedReplica {
                replica_id: (*id).to_string(),
                score: 1.0 - (i as f32) * 0.1,
                matched_tokens: 0,
            })
            .collect()
    }

    #[test]
    fn matches_by_url() {
        let available = vec![
            worker("http://w0:8000"),
            worker("http://w1:8000"),
            worker("http://w2:8000"),
        ];
        let hint = ranked(&["http://w1:8000"]);
        assert_eq!(
            resolve_replica_to_available_index(&hint, &available),
            Some(1)
        );
    }

    #[test]
    fn walks_ranking_best_first() {
        // Best replica has no matching worker; the second-ranked one does.
        let available = vec![worker("http://w0:8000"), worker("http://w1:8000")];
        let hint = ranked(&["http://unknown:8000", "http://w0:8000"]);
        assert_eq!(
            resolve_replica_to_available_index(&hint, &available),
            Some(0)
        );
    }

    #[test]
    fn matches_by_replica_label() {
        let available = vec![
            worker("http://w0:8000"),
            worker_with_replica_label("http://w1:8000", "replica-xyz"),
        ];
        let hint = ranked(&["replica-xyz"]);
        assert_eq!(
            resolve_replica_to_available_index(&hint, &available),
            Some(1)
        );
    }

    #[test]
    fn no_match_returns_none() {
        let available = vec![worker("http://w0:8000"), worker("http://w1:8000")];
        let hint = ranked(&["http://nowhere:8000", "replica-nope"]);
        assert_eq!(resolve_replica_to_available_index(&hint, &available), None);
    }

    #[test]
    fn empty_ranking_returns_none() {
        let available = vec![worker("http://w0:8000")];
        assert_eq!(resolve_replica_to_available_index(&[], &available), None);
    }
}
