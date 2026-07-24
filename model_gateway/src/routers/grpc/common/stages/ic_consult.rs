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

use ic_client::{reason_code, IcClient, IcClientError, RankedReplica};
use tracing::debug;

use crate::{
    config::IcLookupConfig,
    observability::metrics::{metrics_labels, Metrics},
    worker::Worker,
};

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
                // A response arrived but carries no usable replica: NO_HINT /
                // empty ranking, or a server-side TIMEOUT reason (distinct from a
                // client-side deadline, handled in the `Err` arm below). This
                // consult definitively falls back to the policy.
                Metrics::record_ic_consult_latency(hint.lookup_latency_us);
                let outcome = if hint.reason_code == reason_code::TIMEOUT {
                    metrics_labels::IC_CONSULT_TIMEOUT
                } else {
                    metrics_labels::IC_CONSULT_MISS
                };
                Metrics::record_ic_consult(outcome);
                Metrics::record_ic_routing_decision(metrics_labels::IC_DECISION_FELL_BACK);
                debug!(
                    model_id,
                    reason = %hint.reason_code,
                    "IC lookup returned no usable hint; using policy"
                );
                None
            }
            Ok(hint) => {
                // A usable hint was returned. Whether it is actually obeyed
                // depends on the replica-identity resolution at the call site
                // (`select_single_worker`), which records the terminal
                // hit/miss outcome and routing decision. Here we only record the
                // observed server-side lookup latency.
                Metrics::record_ic_consult_latency(hint.lookup_latency_us);
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
                // A client-side deadline is a timeout; every other transport /
                // RPC failure is an error. Both fail open to the policy.
                let outcome = match err {
                    IcClientError::DeadlineExceeded(_) => metrics_labels::IC_CONSULT_TIMEOUT,
                    _ => metrics_labels::IC_CONSULT_ERROR,
                };
                Metrics::record_ic_consult(outcome);
                Metrics::record_ic_routing_decision(metrics_labels::IC_DECISION_FELL_BACK);
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
/// ## Replica-identity seam
///
/// IC returns an engine-defined `replica_id`; SMG workers are keyed by URL. To
/// obey a hint we match `replica_id` against a worker by, in order:
///   1. the worker URL (exact),
///   2. an explicit `replica_id` worker label (exact), then
///   3. the leading host label of the worker URL.
///
/// Rule 3 is the identity contract with the cache plane. When engines are
/// addressed by per-pod DNS — a headless Service / StatefulSet, so replica `i`
/// has a URL like `grpc://<pod-name>.<svc>.<ns>.svc.cluster.local:<port>` — that
/// leading label IS the pod name, which is exactly what the Inference Cache
/// subscriber advertises as its `replica_id` (`--replica-id=$(POD_NAME)`). So a
/// composition layer that addresses workers per-pod gets obeyed routing with no
/// extra identity plumbing; rules 1–2 remain for callers that key identity on the
/// full URL or set an explicit label.
///
/// Walks the ranked list best-first, returning the first replica that maps to a
/// worker in `available`. `None` (no match) means "omit the target" so the caller
/// falls back to its policy.
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

/// Whether `worker` is the target named by `replica_id`: exact URL, else an
/// explicit `replica_id` label, else the leading host label of the worker URL
/// (the per-pod-DNS pod name the cache subscriber reports). See
/// [`resolve_replica_to_available_index`] for the identity contract.
fn worker_matches_replica(worker: &Arc<dyn Worker>, replica_id: &str) -> bool {
    let url = worker.url();
    if url == replica_id {
        return true;
    }
    if worker
        .metadata()
        .spec
        .labels
        .get("replica_id")
        .is_some_and(|label| label == replica_id)
    {
        return true;
    }
    host_leading_label(url).is_some_and(|label| label == replica_id)
}

/// The leading DNS label of a worker URL's host, e.g.
/// `grpc://pod-0.svc.ns.svc.cluster.local:9000` -> `pod-0`. With per-pod-DNS
/// addressing this label is the engine pod name — the cache subscriber's
/// `replica_id`. Returns `None` for an IPv6-literal host (bracketed) or an empty
/// label; an IPv4 host yields its first octet, which is harmless as a last-resort
/// match since IC never reports a bare octet as a `replica_id`.
fn host_leading_label(url: &str) -> Option<&str> {
    // Drop any scheme ("grpc://", "http://", ...) then the path/query/fragment,
    // leaving the authority (host[:port]).
    let after_scheme = url.split_once("://").map_or(url, |(_, rest)| rest);
    let authority = after_scheme
        .split(['/', '?', '#'])
        .next()
        .unwrap_or(after_scheme);
    if authority.starts_with('[') {
        return None; // IPv6 literal has no DNS label
    }
    // The leading DNS label ends at the first '.' (domain) or ':' (port).
    let label = authority.split(['.', ':']).next().unwrap_or(authority);
    (!label.is_empty()).then_some(label)
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
    fn matches_by_host_leading_label() {
        // Per-pod DNS addressing: the URL's leading host label is the engine pod
        // name that IC reports as replica_id, so no explicit label is needed.
        let available = vec![
            worker("grpc://qwen3-engine-0.qwen3-engine.ns.svc.cluster.local:9000"),
            worker("grpc://qwen3-engine-1.qwen3-engine.ns.svc.cluster.local:9000"),
        ];
        let hint = ranked(&["qwen3-engine-1"]);
        assert_eq!(
            resolve_replica_to_available_index(&hint, &available),
            Some(1)
        );
    }

    #[test]
    fn host_leading_label_does_not_false_match_ip_worker() {
        // A pod-name hint must not resolve to an IP-addressed worker.
        let available = vec![worker("grpc://10.0.0.1:9000")];
        let hint = ranked(&["qwen3-engine-0"]);
        assert_eq!(resolve_replica_to_available_index(&hint, &available), None);
    }

    #[test]
    fn host_leading_label_extraction() {
        assert_eq!(
            host_leading_label("grpc://pod-0.svc.ns.svc.cluster.local:9000"),
            Some("pod-0")
        );
        assert_eq!(host_leading_label("http://w1:8000"), Some("w1"));
        assert_eq!(host_leading_label("bare-host:9000"), Some("bare-host"));
        assert_eq!(host_leading_label("grpc://[::1]:9000"), None); // IPv6 literal
        assert_eq!(host_leading_label("grpc://"), None); // empty authority
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
