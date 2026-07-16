//! Tonic gRPC client for the Inference Cache (IC) routing control plane.
//!
//! IC is the routing brain: given a request's `token_ids`, the `LookupRoute`
//! RPC returns a ranked list of replicas whose KV caches already hold a prefix
//! match. SMG makes a single, time-boxed `LookupRoute` call and obeys the best
//! usable replica; on any miss / timeout / transport error it falls back to its
//! existing routing policy. The contract is **fail-open** — an empty result
//! (`NO_HINT` or a diagnostic reason code) is always a valid no-op.
//!
//! This crate is transport-only: it owns the vendored `.proto`, the generated
//! tonic client, a bounded-deadline `lookup_route` wrapper, and a small typed
//! view over the response. Mapping a `replica_id` to a concrete worker, and the
//! decision of when to consult IC at all, live in the gateway (`model_gateway`)
//! — see `routers::grpc::common::stages::ic_consult`.
//!
//! Patterned after `crates/grpc_client` (channel profile, generated-code module
//! wrapper, error idioms).

use std::time::Duration;

use tonic::transport::{Channel, Endpoint};
use tracing::debug;

/// Generated protobuf + tonic client/server code for the IC contract.
///
/// The generated code does not follow the workspace lints; scope the relevant
/// allows to this module only (mirrors `crates/grpc_client`).
#[expect(clippy::allow_attributes)]
pub mod proto {
    #![allow(
        clippy::all,
        clippy::pedantic,
        clippy::absolute_paths,
        unused_qualifications
    )]
    tonic::include_proto!("inferencecache.v1alpha1");
}

use proto::{inference_cache_client::InferenceCacheClient, LookupRouteRequest};

/// Known `reason_code` values returned by `LookupRoute` (see the vendored
/// `.proto`). Only the codes SMG branches on are named; any other string is
/// treated as "no usable hint" by [`LookupResult::is_no_hint`].
pub mod reason_code {
    /// A prefix match was found and `replica_scores` is populated.
    pub const PREFIX_MATCH: &str = "PREFIX_MATCH";
    /// Tenant-affinity hint (no exact prefix match).
    pub const TENANT_HOT: &str = "TENANT_HOT";
    /// Generic affinity hint.
    pub const AFFINITY_HINT: &str = "AFFINITY_HINT";
    /// No hint available — route as normal.
    pub const NO_HINT: &str = "NO_HINT";
    /// Server-side deadline elapsed before a hint could be produced.
    pub const TIMEOUT: &str = "TIMEOUT";
}

/// Errors surfaced by [`IcClient`]. All variants are non-fatal for the caller:
/// the IC lookup is advisory, so every error maps to "route as normal".
#[derive(Debug, thiserror::Error)]
pub enum IcClientError {
    /// The endpoint string could not be parsed into a URI.
    #[error("invalid IC endpoint {endpoint:?}: {message}")]
    InvalidEndpoint { endpoint: String, message: String },
    /// The (eager) TCP / HTTP2 connection to IC failed.
    #[error("failed to connect to IC at {endpoint:?}: {source}")]
    Connect {
        endpoint: String,
        source: tonic::transport::Error,
    },
    /// The `LookupRoute` RPC returned a gRPC error status.
    #[error("IC lookup_route RPC failed: {0}")]
    Rpc(#[from] tonic::Status),
    /// The client-side deadline elapsed before the RPC returned. This is the
    /// hard upper bound that keeps IC off the critical path.
    #[error("IC lookup_route exceeded {0:?} deadline")]
    DeadlineExceeded(Duration),
}

/// A single ranked replica from a `LookupRoute` response. `score` is
/// higher-is-better and the server returns replicas already ordered by score.
#[derive(Clone, Debug, PartialEq)]
pub struct RankedReplica {
    /// Engine-defined replica identity (opaque to SMG; see the replica-identity
    /// seam in `model_gateway`).
    pub replica_id: String,
    /// Server-assigned score; higher is better.
    pub score: f32,
    /// Number of leading tokens the replica already has cached.
    pub matched_tokens: i32,
}

/// Typed view over a `LookupRoute` response.
#[derive(Clone, Debug, Default)]
pub struct LookupResult {
    /// Replicas ranked best-first. Empty on a miss.
    pub replicas: Vec<RankedReplica>,
    /// Raw `reason_code` string from the server.
    pub reason_code: String,
    /// Server-measured lookup latency, in microseconds.
    pub lookup_latency_us: i64,
}

impl LookupResult {
    /// True when the response carries no usable routing hint — an empty ranking
    /// or an explicit `NO_HINT`. Callers treat this as a no-op and fall back to
    /// their default routing policy.
    pub fn is_no_hint(&self) -> bool {
        self.replicas.is_empty() || self.reason_code == reason_code::NO_HINT
    }

    /// The best (highest-scored) replica, if any.
    pub fn best(&self) -> Option<&RankedReplica> {
        self.replicas.first()
    }
}

impl From<proto::LookupRouteResponse> for LookupResult {
    fn from(resp: proto::LookupRouteResponse) -> Self {
        let replicas = resp
            .replica_scores
            .into_iter()
            .map(|s| RankedReplica {
                replica_id: s.replica_id,
                score: s.score,
                matched_tokens: s.matched_tokens,
            })
            .collect();
        Self {
            replicas,
            reason_code: resp.reason_code,
            lookup_latency_us: resp.lookup_latency_us,
        }
    }
}

/// Convert a `grpc://` / `grpcs://` endpoint to a tonic-compatible `http(s)://`
/// URI; other schemes (or schemeless inputs) pass through unchanged. Mirrors
/// `smg_grpc_client::normalize_grpc_endpoint`.
fn normalize_grpc_endpoint(endpoint: &str) -> String {
    match endpoint.split_once("://") {
        Some(("grpc", rest)) => format!("http://{rest}"),
        Some(("grpcs", rest)) => format!("https://{rest}"),
        _ => endpoint.to_string(),
    }
}

/// A cloneable gRPC client for the IC `InferenceCache` service.
///
/// Cloning is cheap — the underlying `tonic::Channel` is reference-counted and
/// multiplexes over a single HTTP/2 connection.
#[derive(Clone, Debug)]
pub struct IcClient {
    inner: InferenceCacheClient<Channel>,
    endpoint: String,
}

impl IcClient {
    /// Eagerly connect, awaiting the TCP + HTTP/2 handshake (bounded by
    /// `connect_timeout`). Prefer [`IcClient::connect_lazy`] on the router
    /// startup path so an unreachable IC never blocks or fails boot; this
    /// constructor is mainly for tests and health-gated callers.
    pub async fn connect(
        endpoint: impl Into<String>,
        connect_timeout: Duration,
    ) -> Result<Self, IcClientError> {
        let endpoint = endpoint.into();
        let ep = build_endpoint(&endpoint, connect_timeout)?;
        let channel = ep
            .connect()
            .await
            .map_err(|source| IcClientError::Connect {
                endpoint: endpoint.clone(),
                source,
            })?;
        debug!(endpoint = %endpoint, "Connected to IC gRPC service");
        Ok(Self {
            inner: InferenceCacheClient::new(channel),
            endpoint,
        })
    }

    /// Build a lazily-connecting client: the channel dials on the first RPC, so
    /// construction never blocks and only fails on a malformed URI. This is the
    /// production path — IC being down must never fail router startup, and a
    /// failed dial surfaces as a per-request (fail-open) RPC error instead.
    pub fn connect_lazy(
        endpoint: impl Into<String>,
        connect_timeout: Duration,
    ) -> Result<Self, IcClientError> {
        let endpoint = endpoint.into();
        let ep = build_endpoint(&endpoint, connect_timeout)?;
        let channel = ep.connect_lazy();
        debug!(endpoint = %endpoint, "Created lazy IC gRPC channel");
        Ok(Self {
            inner: InferenceCacheClient::new(channel),
            endpoint,
        })
    }

    /// The endpoint this client was constructed with (for logging).
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Look up the best replica(s) for a request's `token_ids`, bounded by
    /// `deadline`.
    ///
    /// The deadline is a hard client-side `tokio::time::timeout`: the call never
    /// outlives its budget, and abandoning the future cancels the in-flight RPC
    /// (HTTP/2 reset) so the server stops working on it. A miss is **not** an
    /// error — it returns `Ok` with an empty [`LookupResult`]; see
    /// [`LookupResult::is_no_hint`]. Only transport failures and an elapsed
    /// deadline produce `Err`, and all of those are advisory for the caller
    /// (fail-open: route as normal).
    pub async fn lookup_route(
        &self,
        model_id: impl Into<String>,
        tenant_id: impl Into<String>,
        hash_scheme: impl Into<String>,
        token_ids: Vec<u32>,
        deadline: Duration,
    ) -> Result<LookupResult, IcClientError> {
        let request = LookupRouteRequest {
            model_id: model_id.into(),
            tenant_id: tenant_id.into(),
            hash_scheme: hash_scheme.into(),
            token_ids,
            ..Default::default()
        };

        let mut client = self.inner.clone();
        let rpc = client.lookup_route(tonic::Request::new(request));

        match tokio::time::timeout(deadline, rpc).await {
            Ok(Ok(resp)) => Ok(LookupResult::from(resp.into_inner())),
            Ok(Err(status)) => Err(IcClientError::Rpc(status)),
            Err(_elapsed) => Err(IcClientError::DeadlineExceeded(deadline)),
        }
    }
}

/// Build a tonic `Endpoint` from a (possibly `grpc://`) endpoint string with the
/// IC connect profile applied.
fn build_endpoint(endpoint: &str, connect_timeout: Duration) -> Result<Endpoint, IcClientError> {
    let http = normalize_grpc_endpoint(endpoint);
    let ep = Channel::from_shared(http)
        .map_err(|err| IcClientError::InvalidEndpoint {
            endpoint: endpoint.to_string(),
            message: err.to_string(),
        })?
        .connect_timeout(connect_timeout)
        .tcp_nodelay(true);
    Ok(ep)
}

#[cfg(test)]
mod tests {
    use std::{net::SocketAddr, time::Duration};

    use futures::stream;
    use tonic::{transport::Server, Request, Response, Status};

    use super::{
        proto::{
            inference_cache_server::{InferenceCache, InferenceCacheServer},
            Ack, CacheEvent, CacheStateUpdate, GetCacheStateRequest, GetCacheStateResponse,
            LookupPdRouteRequest, LookupPdRouteResponse, LookupRouteRequest, LookupRouteResponse,
            Metric, RenderTemplateRequest, RenderTemplateResponse, ReplicaScore,
            StreamEventsRequest, StreamMetricsRequest,
        },
        IcClient, IcClientError,
    };

    /// In-process mock IC server. Behavior is keyed off `model_id` so a single
    /// implementation covers every test case.
    #[derive(Default)]
    struct MockIc;

    #[tonic::async_trait]
    impl InferenceCache for MockIc {
        async fn lookup_route(
            &self,
            request: Request<LookupRouteRequest>,
        ) -> Result<Response<LookupRouteResponse>, Status> {
            let req = request.into_inner();
            match req.model_id.as_str() {
                // Sleep well past any sane client deadline to exercise the
                // fail-open timeout path.
                "slow-model" => {
                    tokio::time::sleep(Duration::from_secs(30)).await;
                    Ok(Response::new(LookupRouteResponse::default()))
                }
                // Explicit miss: empty ranking + NO_HINT.
                "no-hint-model" => Ok(Response::new(LookupRouteResponse {
                    replica_scores: vec![],
                    reason_code: super::reason_code::NO_HINT.to_string(),
                    lookup_latency_us: 7,
                    token_ids: vec![],
                })),
                // Happy path: two ranked replicas, best-first.
                _ => Ok(Response::new(LookupRouteResponse {
                    replica_scores: vec![
                        ReplicaScore {
                            replica_id: "replica-a".to_string(),
                            score: 0.9,
                            matched_tokens: 128,
                            estimated_cache_hit_prob: 0.8,
                        },
                        ReplicaScore {
                            replica_id: "replica-b".to_string(),
                            score: 0.4,
                            matched_tokens: 32,
                            estimated_cache_hit_prob: 0.3,
                        },
                    ],
                    reason_code: super::reason_code::PREFIX_MATCH.to_string(),
                    lookup_latency_us: 42,
                    token_ids: vec![],
                })),
            }
        }

        // Remaining RPCs are unused by the client; stub them out.
        async fn render_template(
            &self,
            _request: Request<RenderTemplateRequest>,
        ) -> Result<Response<RenderTemplateResponse>, Status> {
            Err(Status::unimplemented("render_template"))
        }

        async fn lookup_pd_route(
            &self,
            _request: Request<LookupPdRouteRequest>,
        ) -> Result<Response<LookupPdRouteResponse>, Status> {
            Err(Status::unimplemented("lookup_pd_route"))
        }

        async fn get_cache_state(
            &self,
            _request: Request<GetCacheStateRequest>,
        ) -> Result<Response<GetCacheStateResponse>, Status> {
            Err(Status::unimplemented("get_cache_state"))
        }

        async fn report_cache_state(
            &self,
            _request: Request<tonic::Streaming<CacheStateUpdate>>,
        ) -> Result<Response<Ack>, Status> {
            Err(Status::unimplemented("report_cache_state"))
        }

        async fn publish_event(
            &self,
            _request: Request<CacheEvent>,
        ) -> Result<Response<Ack>, Status> {
            Err(Status::unimplemented("publish_event"))
        }

        type StreamCacheEventsStream = stream::Empty<Result<CacheEvent, Status>>;

        async fn stream_cache_events(
            &self,
            _request: Request<StreamEventsRequest>,
        ) -> Result<Response<Self::StreamCacheEventsStream>, Status> {
            Err(Status::unimplemented("stream_cache_events"))
        }

        type StreamMetricsStream = stream::Empty<Result<Metric, Status>>;

        async fn stream_metrics(
            &self,
            _request: Request<StreamMetricsRequest>,
        ) -> Result<Response<Self::StreamMetricsStream>, Status> {
            Err(Status::unimplemented("stream_metrics"))
        }
    }

    /// Bind an ephemeral port and serve `MockIc` on it. The listener is bound
    /// before the serve loop is spawned, so the returned address is immediately
    /// connectable (the OS queues the connection until `accept`).
    #[expect(
        clippy::disallowed_methods,
        reason = "test-only mock server; the task lives and dies with the test runtime"
    )]
    async fn spawn_mock() -> SocketAddr {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind ephemeral port");
        let addr = listener.local_addr().expect("local addr");

        // Dependency-free `Incoming` stream over the bound listener.
        let incoming = stream::unfold(listener, |listener| async move {
            let item = listener.accept().await.map(|(stream, _peer)| stream);
            Some((item, listener))
        });

        tokio::spawn(async move {
            Server::builder()
                .add_service(InferenceCacheServer::new(MockIc))
                .serve_with_incoming(incoming)
                .await
                .expect("mock IC server");
        });

        addr
    }

    #[tokio::test]
    async fn happy_path_returns_ranked_replicas() {
        let addr = spawn_mock().await;
        let client = IcClient::connect(format!("http://{addr}"), Duration::from_secs(2))
            .await
            .expect("connect");

        let result = client
            .lookup_route(
                "llama-3-8b",
                "tenant-1",
                "sglang-v1",
                vec![1, 2, 3, 4],
                Duration::from_secs(2),
            )
            .await
            .expect("lookup ok");

        assert!(!result.is_no_hint());
        assert_eq!(result.reason_code, super::reason_code::PREFIX_MATCH);
        assert_eq!(result.replicas.len(), 2);
        let best = result.best().expect("best replica");
        assert_eq!(best.replica_id, "replica-a");
        assert_eq!(best.matched_tokens, 128);
        assert!((best.score - 0.9).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn deadline_exceeded_is_error() {
        let addr = spawn_mock().await;
        let client = IcClient::connect(format!("http://{addr}"), Duration::from_secs(2))
            .await
            .expect("connect");

        let err = client
            .lookup_route(
                "slow-model",
                "tenant-1",
                "sglang-v1",
                vec![9, 9, 9],
                Duration::from_millis(100),
            )
            .await
            .expect_err("should time out");

        match err {
            IcClientError::DeadlineExceeded(d) => assert_eq!(d, Duration::from_millis(100)),
            other => panic!("expected DeadlineExceeded, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn no_hint_passthrough() {
        let addr = spawn_mock().await;
        let client = IcClient::connect(format!("http://{addr}"), Duration::from_secs(2))
            .await
            .expect("connect");

        let result = client
            .lookup_route(
                "no-hint-model",
                "tenant-1",
                "sglang-v1",
                vec![5, 6, 7],
                Duration::from_secs(2),
            )
            .await
            .expect("lookup ok");

        assert!(result.is_no_hint());
        assert!(result.replicas.is_empty());
        assert!(result.best().is_none());
        assert_eq!(result.reason_code, super::reason_code::NO_HINT);
    }

    #[test]
    fn normalize_rewrites_grpc_schemes() {
        assert_eq!(
            super::normalize_grpc_endpoint("grpc://ic:9100"),
            "http://ic:9100"
        );
        assert_eq!(
            super::normalize_grpc_endpoint("grpcs://ic:9443"),
            "https://ic:9443"
        );
        assert_eq!(
            super::normalize_grpc_endpoint("http://ic:9100"),
            "http://ic:9100"
        );
    }
}
